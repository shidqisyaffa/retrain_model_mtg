"""TensorRT runtime for TC-CLIP, with an optional bounded shadow comparison."""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.services.detection.tcclip_prompts import TC_CLIP_MAX_PROMPTS, TcClipPrompts
from app.services.detection.tcclip_scores import aggregate_tcclip_logits
from app.services.detection.tcclip_stable_merge import MERGE_VERSION

logger = logging.getLogger(__name__)
ARTIFACT_VERSION = 1  # legacy prompt-embedded engine
DYNAMIC_ARTIFACT_VERSION = 2
DYNAMIC_MIN_CLASSES = 2
DYNAMIC_OPT_CLASSES = 8
DYNAMIC_MAX_CLASSES = TC_CLIP_MAX_PROMPTS  # Manifest key retained for existing dynamic engines.


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_signature(prompts: TcClipPrompts) -> str:
    rows = [(item.name, variant.prompt) for item in prompts.classes for variant in item.prompts]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_metadata(
    *,
    checkpoint_path: Path,
    prompts: TcClipPrompts,
    num_frames: int,
    batch_size: int,
    trt_version: str,
) -> dict[str, object]:
    """Metadata for older engines that embed the prompt text."""
    return {
        "artifact_version": ARTIFACT_VERSION,
        "merge_version": MERGE_VERSION,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "prompt_signature": prompt_signature(prompts),
        "prompt_names": list(prompts.names),
        "num_frames": num_frames,
        "batch_size": batch_size,
        "image_size": 224,
        "gpu_compute_capability": list(torch.cuda.get_device_capability()),
        "tensorrt_version": trt_version,
    }


def dynamic_artifact_metadata(
    *,
    checkpoint_path: Path,
    num_frames: int,
    batch_size: int,
    trt_version: str,
) -> dict[str, object]:
    """Prompt text is a runtime input; only model and hardware bind the plan."""
    return {
        "artifact_version": DYNAMIC_ARTIFACT_VERSION,
        "merge_version": MERGE_VERSION,
        "prompt_mode": "dynamic",
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "num_frames": num_frames,
        "batch_size": batch_size,
        "image_size": 224,
        "min_classes": DYNAMIC_MIN_CLASSES,
        "opt_classes": DYNAMIC_OPT_CLASSES,
        "max_classes": DYNAMIC_MAX_CLASSES,
        "gpu_compute_capability": list(torch.cuda.get_device_capability()),
        "tensorrt_version": trt_version,
    }


def runner_accepts_prompts(runner: object, prompts: TcClipPrompts) -> bool:
    supports = getattr(runner, "supports_prompts", None)
    if callable(supports):
        return bool(supports(prompts))
    return getattr(runner, "prompt_signature", None) == prompt_signature(prompts)


class TensorRtRunner:
    """Run either a legacy static engine or a prompt-dynamic FP16 engine."""

    def __init__(self, engine_path: Path, *, checkpoint_path: Path, prompts: TcClipPrompts, num_frames: int) -> None:
        import tensorrt as trt

        metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.is_dynamic = metadata.get("artifact_version") == DYNAMIC_ARTIFACT_VERSION
        if self.is_dynamic:
            expected = dynamic_artifact_metadata(
                checkpoint_path=checkpoint_path,
                num_frames=num_frames,
                batch_size=int(metadata["batch_size"]),
                trt_version=trt.__version__,
            )
        else:
            expected = artifact_metadata(
                checkpoint_path=checkpoint_path,
                prompts=prompts,
                num_frames=num_frames,
                batch_size=int(metadata["batch_size"]),
                trt_version=trt.__version__,
            )
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"TensorRT engine metadata mismatch: {key}")
        self.batch_size = int(metadata["batch_size"])
        self.max_classes = DYNAMIC_MAX_CLASSES if self.is_dynamic else prompts.prompt_count
        self._prompts = prompts
        self.prompt_signature = None if self.is_dynamic else str(metadata["prompt_signature"])
        if not self.supports_prompts(prompts):
            raise ValueError("TensorRT engine does not support the active TC-CLIP prompts")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")
        self._runtime = trt.Runtime(trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError("Unable to deserialize TC-CLIP TensorRT engine")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Unable to create TC-CLIP TensorRT context")
        inputs = [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        outputs = [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        if len(outputs) != 1:
            raise ValueError("TC-CLIP TensorRT engine must have one output")
        self._output_name = outputs[0]
        if self.is_dynamic:
            if set(inputs) != {"frames", "prompts", "tokenized_prompts"}:
                raise ValueError(f"Dynamic TC-CLIP engine inputs are invalid: {inputs}")
            if tuple(self._engine.get_tensor_shape("prompts")) != (-1, 30, 512):
                raise ValueError("Dynamic TC-CLIP prompt embedding shape is invalid")
            if tuple(self._engine.get_tensor_shape("tokenized_prompts")) != (-1, 30):
                raise ValueError("Dynamic TC-CLIP token shape is invalid")
            self._input_name = "frames"
            prompt_dtype = self._engine.get_tensor_dtype("prompts")
            token_dtype = self._engine.get_tensor_dtype("tokenized_prompts")
            if prompt_dtype not in (trt.float16, trt.float32) or token_dtype not in (trt.int32, trt.int64):
                raise ValueError("Dynamic TC-CLIP prompt input dtypes are unsupported")
            self._prompt_dtype = torch.float16 if prompt_dtype == trt.float16 else torch.float32
            self._token_dtype = torch.int32 if token_dtype == trt.int32 else torch.int64
        else:
            if len(inputs) != 1:
                raise ValueError("Legacy TC-CLIP TensorRT engine must have one input")
            self._input_name = inputs[0]
        if tuple(self._engine.get_tensor_shape(self._input_name)) != (self.batch_size, num_frames, 3, 224, 224):
            raise ValueError("TensorRT engine input shape does not match the configured TC-CLIP model")
        output_dtype = self._engine.get_tensor_dtype(self._output_name)
        if output_dtype not in (trt.float16, trt.float32):
            raise ValueError(f"Unsupported TensorRT output dtype: {output_dtype}")
        self._output_dtype = torch.float16 if output_dtype == trt.float16 else torch.float32
        self._stream = torch.cuda.Stream()
        self._lock = threading.RLock()

    def supports_prompts(self, prompts: TcClipPrompts) -> bool:
        if self.is_dynamic:
            return DYNAMIC_MIN_CLASSES <= prompts.prompt_count <= self.max_classes
        return self.prompt_signature == prompt_signature(prompts)

    def predict(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        prompt_embeddings: torch.Tensor | None = None,
        tokenized_prompts: torch.Tensor | None = None,
        prompt_classes: TcClipPrompts | None = None,
    ) -> np.ndarray:
        if not tensors:
            return np.empty((0, 0), dtype=np.float32)
        active_prompts = prompt_classes or self._prompts
        if not self.supports_prompts(active_prompts):
            raise ValueError("TensorRT engine does not support the active TC-CLIP prompts")
        if self.is_dynamic:
            if prompt_embeddings is None or tokenized_prompts is None:
                raise ValueError("Dynamic TensorRT requires current prompt embeddings and token IDs")
            count = int(prompt_embeddings.shape[0])
            if count != active_prompts.prompt_count:
                raise ValueError("TensorRT prompt embeddings do not match the active class-to-prompt mapping")
            if not DYNAMIC_MIN_CLASSES <= count <= self.max_classes:
                raise ValueError(f"Dynamic TensorRT supports {DYNAMIC_MIN_CLASSES}-{self.max_classes} text candidates")
            if tuple(prompt_embeddings.shape[1:]) != (30, 512) or tuple(tokenized_prompts.shape) != (count, 30):
                raise ValueError("Dynamic TensorRT prompt inputs have incorrect shapes")
        elif prompt_embeddings is not None or tokenized_prompts is not None:
            raise ValueError("Legacy TensorRT engine embeds its prompts")

        rows: list[np.ndarray] = []
        with self._lock:
            # Prompt buffers may have just been produced on PyTorch's current stream.
            self._stream.wait_stream(torch.cuda.current_stream())
            for offset in range(0, len(tensors), self.batch_size):
                chunk = tensors[offset : offset + self.batch_size]
                padded = chunk + (chunk[-1],) * (self.batch_size - len(chunk))
                with torch.cuda.stream(self._stream):
                    model_input = torch.stack(padded).to("cuda", non_blocking=True)
                    if self.is_dynamic:
                        assert prompt_embeddings is not None and tokenized_prompts is not None
                        prompt_input = prompt_embeddings.to(
                            device="cuda", dtype=self._prompt_dtype, non_blocking=True
                        ).contiguous()
                        token_input = tokenized_prompts.to(
                            device="cuda", dtype=self._token_dtype, non_blocking=True
                        ).contiguous()
                        if not self._context.set_input_shape("prompts", tuple(prompt_input.shape)):
                            raise RuntimeError("TensorRT rejected the prompt embedding shape")
                        if not self._context.set_input_shape("tokenized_prompts", tuple(token_input.shape)):
                            raise RuntimeError("TensorRT rejected the prompt token shape")
                        self._context.set_tensor_address("prompts", prompt_input.data_ptr())
                        self._context.set_tensor_address("tokenized_prompts", token_input.data_ptr())
                    shape = tuple(self._context.get_tensor_shape(self._output_name))
                    if any(size < 1 for size in shape):
                        raise RuntimeError(f"TensorRT returned an unresolved output shape: {shape}")
                    model_output = torch.empty(shape, dtype=self._output_dtype, device="cuda")
                    self._context.set_tensor_address(self._input_name, model_input.data_ptr())
                    self._context.set_tensor_address(self._output_name, model_output.data_ptr())
                    if not self._context.execute_async_v3(self._stream.cuda_stream):
                        raise RuntimeError("TensorRT TC-CLIP execution failed")
                    class_logits = aggregate_tcclip_logits(model_output[: len(chunk)].float(), active_prompts)
                    probabilities = torch.softmax(class_logits, dim=1)
                self._stream.synchronize()
                rows.append(probabilities.cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0)


@dataclass(slots=True)
class _ShadowJob:
    tensors: tuple[torch.Tensor, ...]
    reference: np.ndarray
    prompts: TcClipPrompts


class TcClipTensorRtShadow:
    """Sample at most one queued CCTV batch without delaying the primary result."""

    def __init__(
        self,
        *,
        engine_path: Path,
        checkpoint_path: Path,
        prompts: TcClipPrompts,
        num_frames: int,
        sample_every: int,
        threshold: float,
    ) -> None:
        self.engine_path = engine_path
        self.checkpoint_path = checkpoint_path
        self.prompts = prompts
        self.num_frames = num_frames
        self.sample_every = max(1, sample_every)
        self.threshold = threshold
        self._queue: queue.Queue[_ShadowJob | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._observed = 0
        self._compared = 0
        self._skipped_busy = 0
        self._skipped_prompts = 0
        self._decision_mismatches = 0
        self._top_class_mismatches = 0
        self._max_score_delta = 0.0
        self._last_ms = 0.0
        self._error: str | None = None
        self._disabled = False
        self._ready = False
        self._worker = threading.Thread(target=self._run, name="tcclip-tensorrt-shadow", daemon=True)
        self._worker.start()

    def submit(self, tensors: list[torch.Tensor], reference: np.ndarray, prompts: TcClipPrompts) -> None:
        with self._lock:
            if self._disabled:
                return
            self._observed += 1
            if self._observed % self.sample_every:
                return
            if self._queue.full():
                self._skipped_busy += 1
                return
            try:
                self._queue.put_nowait(_ShadowJob(tuple(tensors), reference.copy(), prompts))
            except queue.Full:
                self._skipped_busy += 1

    def _run(self) -> None:
        try:
            runner = TensorRtRunner(
                self.engine_path,
                checkpoint_path=self.checkpoint_path,
                prompts=self.prompts,
                num_frames=self.num_frames,
            )
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._disabled = True
            logger.warning("TC-CLIP TensorRT shadow disabled: %s", exc)
            return
        with self._lock:
            self._ready = True
        while True:
            job = self._queue.get()
            if job is None:
                return
            if prompt_signature(job.prompts) != runner.prompt_signature:
                with self._lock:
                    self._skipped_prompts += 1
                continue
            try:
                started = time.perf_counter()
                result = runner.predict(job.tensors)
                elapsed_ms = (time.perf_counter() - started) * 1000
                if result.shape != job.reference.shape:
                    raise ValueError("TensorRT shadow result shape differs from PyTorch")
                reference_top = np.argmax(job.reference, axis=1)
                result_top = np.argmax(result, axis=1)
                reference_alert = np.array(
                    [
                        job.prompts.classes[i].is_alert and job.reference[row, i] >= self.threshold
                        for row, i in enumerate(reference_top)
                    ]
                )
                result_alert = np.array(
                    [
                        job.prompts.classes[i].is_alert and result[row, i] >= self.threshold
                        for row, i in enumerate(result_top)
                    ]
                )
                with self._lock:
                    self._compared += len(result)
                    self._top_class_mismatches += int(np.count_nonzero(reference_top != result_top))
                    self._decision_mismatches += int(np.count_nonzero(reference_alert != result_alert))
                    self._max_score_delta = max(self._max_score_delta, float(np.max(np.abs(job.reference - result))))
                    self._last_ms = round(elapsed_ms, 2)
            except Exception as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._disabled = True
                logger.warning("TC-CLIP TensorRT shadow disabled after inference failure: %s", exc)
                return

    def get_info(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": not self._disabled,
                "ready": self._ready,
                "observed_batches": self._observed,
                "compared_clips": self._compared,
                "skipped_busy": self._skipped_busy,
                "skipped_prompt_mismatch": self._skipped_prompts,
                "top_class_mismatches": self._top_class_mismatches,
                "alert_decision_mismatches": self._decision_mismatches,
                "max_score_delta": round(self._max_score_delta, 6),
                "last_batch_ms": self._last_ms,
                "last_error": self._error,
                "engine_path": str(self.engine_path),
            }

    def close(self) -> None:
        with self._lock:
            self._disabled = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            with suppress(queue.Empty):
                self._queue.get_nowait()
            self._queue.put_nowait(None)
        self._worker.join(timeout=2)
