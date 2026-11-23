from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
import weakref
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.services.detection.tcclip_prompt_tokens import (
    TC_CLIP_CONTEXT_LENGTH,
    TC_CLIP_PROMPT_PREFIX,
    validate_tcclip_prompt_tokens,
)
from app.services.detection.tcclip_prompts import (
    TC_CLIP_MAX_CLASSES as TC_CLIP_MAX_CLASSES,
)
from app.services.detection.tcclip_prompts import (
    TcClipPromptClass as TcClipPromptClass,
)
from app.services.detection.tcclip_prompts import (
    TcClipPrompts as TcClipPrompts,
)
from app.services.detection.tcclip_scores import aggregate_tcclip_logits

logger = logging.getLogger(__name__)

TC_CLIP_OUTPUT_NAMES = ("normal", "fight")
TC_CLIP_IMAGE_SIZE = 224


def _module_runtime_locations(module: Any) -> tuple[Path, ...]:
    """Return filesystem locations for regular and namespace packages."""
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(Path(module_file).resolve())
    for module_path in getattr(module, "__path__", ()):
        locations.append(Path(module_path).resolve())
    return tuple(locations)


def _module_belongs_to_runtime(module: Any, runtime_root: Path) -> bool:
    return any(
        location == runtime_root or runtime_root in location.parents for location in _module_runtime_locations(module)
    )


class TcClipActionClassifier:
    """Local TC-CLIP zero-shot classifier for temporal BGR frame crops."""

    def __init__(
        self,
        *,
        runtime_root: str | Path,
        checkpoint_path: str | Path,
        cache_dir: str | Path,
        prompts: TcClipPrompts,
        device: torch.device,
        num_frames: int = 16,
        batch_size: int = 1,
        stable_merge: bool = False,
        shadow_engine_path: str | Path | None = None,
        shadow_sample_every: int = 20,
        shadow_threshold: float = 0.70,
    ) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.device = device
        self.num_frames = int(num_frames)
        self.batch_size = int(batch_size)
        self.stable_merge = bool(stable_merge)
        self.shadow_engine_path = Path(shadow_engine_path).expanduser().resolve() if shadow_engine_path else None
        self.shadow_sample_every = max(1, int(shadow_sample_every))
        self.shadow_threshold = float(shadow_threshold)
        self._shadow: Any = None
        self._shadow_lock = threading.Lock()
        self._shadow_error: str | None = None
        self.prompts = prompts

        if self.num_frames < 2:
            raise ValueError("TC-CLIP num_frames must be at least 2")
        if self.batch_size < 1:
            raise ValueError("TC-CLIP batch_size must be at least 1")
        self._validate_paths()
        validate_tcclip_prompt_tokens(self.prompts.to_payload(), runtime_root=self.runtime_root)

        self._transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(TC_CLIP_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._inference_lock = threading.RLock()
        self._safe_batch_size: int | None = None
        self._oom_retries = 0
        self._last_inference: dict[str, Any] = {}
        self._config: Any = None
        self._model: Any = None
        self._clip_model: Any = None
        self._load_model()

    def _validate_paths(self) -> None:
        required_runtime_dirs = ("clip", "trainers", "utils")
        if not self.runtime_root.is_dir():
            raise FileNotFoundError(f"TC-CLIP runtime directory not found: {self.runtime_root}")
        missing = [name for name in required_runtime_dirs if not (self.runtime_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"TC-CLIP runtime is incomplete at {self.runtime_root}; missing: {', '.join(missing)}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"TC-CLIP checkpoint not found: {self.checkpoint_path}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _import_runtime(self) -> tuple[Any, Any, Any]:
        runtime_text = str(self.runtime_root)
        for module_name in ("clip", "trainers", "utils"):
            loaded = sys.modules.get(module_name)
            if loaded is not None and not _module_belongs_to_runtime(loaded, self.runtime_root):
                locations = _module_runtime_locations(loaded)
                origin = ", ".join(str(location) for location in locations) or "<unknown>"
                raise RuntimeError(
                    f"Python module '{module_name}' was already imported from {origin}; "
                    f"TC-CLIP requires the runtime under {self.runtime_root}"
                )

        if runtime_text not in sys.path:
            sys.path.insert(0, runtime_text)

        try:
            omega_conf = importlib.import_module("omegaconf").OmegaConf
            return_clip = importlib.import_module("trainers.build_trainer").returnCLIP
            load_checkpoint = importlib.import_module("utils.tools").load_checkpoint
        except ImportError as exc:
            raise RuntimeError(
                "TC-CLIP dependencies are incomplete. Run 'uv sync' and verify the configured runtime root."
            ) from exc
        return omega_conf, return_clip, load_checkpoint

    def _load_model(self) -> None:
        omega_conf, return_clip, load_checkpoint = self._import_runtime()
        if self.stable_merge:
            from app.services.detection.tcclip_stable_merge import install_stable_merge

            install_stable_merge()
        os.environ.setdefault("TC_CLIP_CACHE", str(self.cache_dir))
        self._config = omega_conf.create(
            {
                "trainer_name": "TCCLIP",
                "model_arch": "ViT-B/16",
                "vision_model": "TCVisionTransformer",
                "vision_block": "TCAttentionBlock",
                "text_block": "ResidualAttentionBlock",
                "use_custom_attention": True,
                "num_frames": self.num_frames,
                "positional_embedding_type": "space",
                "local_global_bias": True,
                "context_token_k": 96,
                "seed_token_a": 0.3,
                "tome_r": 100,
                "tome_d": 0,
                "n_ctx": 4,
                "ctx_init": TC_CLIP_PROMPT_PREFIX,
                "im_stop_grad": True,
                "prompt_generation_layer_init": "clip",
                "prompt_generation_layer_level": 12,
                "context_length": TC_CLIP_CONTEXT_LENGTH,
                "opt_level": "O1" if self.device.type == "cuda" else "O0",
                "parse_ucf101": False,
                "resume": str(self.checkpoint_path),
                "freeze": None,
                "wise_ft": 0.0,
            }
        )

        self._model, self._clip_model = return_clip(
            self._config,
            logger,
            list(self.prompts.values),
            return_clip_model=True,
        )
        self._model.to(self.device).eval()
        load_checkpoint(
            config=self._config,
            model=self._model,
            optimizer=None,
            lr_scheduler=None,
            logger=logger,
            model_only=True,
        )
        logger.info(
            "TC-CLIP loaded checkpoint='%s' prompts=%s frames=%d device=%s",
            self.checkpoint_path,
            self.prompts.names,
            self.num_frames,
            self.device,
        )

    def update_prompts(
        self,
        prompts: TcClipPrompts | str,
        fight_prompt: str | None = None,
    ) -> bool:
        """Replace dynamic prompt candidates without reloading model weights."""
        if not isinstance(prompts, TcClipPrompts):
            prompts = TcClipPrompts(prompts, fight_prompt)
        validate_tcclip_prompt_tokens(prompts.to_payload(), runtime_root=self.runtime_root)
        with self._inference_lock:
            if self.prompts == prompts:
                return False
            if self.prompts.values == prompts.values:
                # Source/translation provenance, class mapping and alert policy
                # do not change encoded text. Retain identical prompt buffers
                # rather than rebuilding them (and introducing dtype rounding).
                self.prompts = prompts
                return True
            self._model._rebuild_classnames(
                self._config,
                list(prompts.values),
                self._clip_model,
                logger,
            )
            self._model.to(self.device).eval()
            self.prompts = prompts
            return True

    def _build_clip_tensor(self, clip: Sequence[np.ndarray]) -> torch.Tensor:
        if len(clip) != self.num_frames:
            raise ValueError(f"TC-CLIP expects {self.num_frames} frames, got {len(clip)}")
        frames: list[torch.Tensor] = []
        for frame_bgr in clip:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(self._transform(Image.fromarray(frame_rgb)))
        return torch.stack(frames, dim=0)

    def predict(self, clips: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        """Return probabilities ordered according to the active prompt classes."""
        probabilities, _ = self.predict_with_prompts(clips)
        return probabilities

    def predict_with_prompts(
        self,
        clips: Sequence[Sequence[np.ndarray]],
        *,
        batch_size: int | None = None,
        shadow: bool = False,
        tensorrt_runner: Any = None,
    ) -> tuple[np.ndarray, TcClipPrompts]:
        """Return probabilities together with the immutable prompt snapshot used."""
        if not clips:
            with self._inference_lock:
                prompts = self.prompts
            return np.empty((0, len(prompts.classes)), dtype=np.float32), prompts

        requested_batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))
        started_at = time.perf_counter()
        tensors = [self._build_clip_tensor(clip) for clip in clips]
        prepared_at = time.perf_counter()
        chunks: list[np.ndarray] = []
        with self._inference_lock, torch.inference_mode():
            locked_at = time.perf_counter()
            prompts = self.prompts
            if tensorrt_runner is not None:
                from app.services.detection.tcclip_tensorrt_shadow import runner_accepts_prompts

                if not runner_accepts_prompts(tensorrt_runner, prompts):
                    raise ValueError("TensorRT engine prompts differ from the active TC-CLIP prompts")
                if getattr(tensorrt_runner, "is_dynamic", False):
                    result = tensorrt_runner.predict(
                        tuple(tensors),
                        prompt_embeddings=self._model.prompt_learner(),
                        tokenized_prompts=self._model.tokenized_prompts,
                        prompt_classes=prompts,
                    )
                else:
                    result = tensorrt_runner.predict(tuple(tensors))
                if result.shape != (len(tensors), len(prompts.classes)):
                    raise RuntimeError(f"TensorRT returned an unexpected probability shape: {result.shape}")
                completed_at = time.perf_counter()
                self._last_inference = {
                    "engine": "tensorrt",
                    "clips": len(clips),
                    "frames_per_clip": self.num_frames,
                    "requested_batch_size": requested_batch_size,
                    "forward_batch_sizes": [
                        min(tensorrt_runner.batch_size, len(tensors) - offset)
                        for offset in range(0, len(tensors), tensorrt_runner.batch_size)
                    ],
                    "preprocess_ms": round((prepared_at - started_at) * 1000, 2),
                    "lock_wait_ms": round((locked_at - prepared_at) * 1000, 2),
                    "model_and_transfer_ms": round((completed_at - locked_at) * 1000, 2),
                    "total_ms": round((completed_at - started_at) * 1000, 2),
                }
                return result, prompts
            start = 0
            forward_sizes = []
            while start < len(tensors):
                safe_limit = getattr(self, "_safe_batch_size", None)
                effective_batch_size = min(requested_batch_size, safe_limit or requested_batch_size)
                chunk = tensors[start : start + effective_batch_size]
                retry = False
                try:
                    probabilities = self._forward_tensor_batch(chunk, len(prompts.classes))
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda" or len(chunk) == 1:
                        raise
                    self._safe_batch_size = max(1, len(chunk) // 2)
                    self._oom_retries = getattr(self, "_oom_retries", 0) + 1
                    logger.warning("TC-CLIP CUDA batch exhausted memory; retrying with %d clips", self._safe_batch_size)
                    retry = True
                if retry:
                    # Clear the failed forward's traceback before releasing
                    # cached allocations; retry the same ROI rows in order.
                    torch.cuda.empty_cache()
                    continue
                chunks.append(probabilities)
                forward_sizes.append(len(chunk))
                start += len(chunk)
            completed_at = time.perf_counter()
            self._last_inference = {
                "engine": "pytorch",
                "clips": len(clips),
                "frames_per_clip": getattr(self, "num_frames", None),
                "requested_batch_size": requested_batch_size,
                "forward_batch_sizes": forward_sizes,
                "preprocess_ms": round((prepared_at - started_at) * 1000, 2),
                "lock_wait_ms": round((locked_at - prepared_at) * 1000, 2),
                # Includes host/device transfer, forward, softmax and result
                # copy. The CPU copy synchronizes CUDA before this measurement.
                "model_and_transfer_ms": round((completed_at - locked_at) * 1000, 2),
                "total_ms": round((completed_at - started_at) * 1000, 2),
            }
        result = np.concatenate(chunks, axis=0)
        if shadow and self.stable_merge and self.device.type == "cuda" and self.shadow_engine_path:
            try:
                with self._shadow_lock:
                    if self._shadow is None and self._shadow_error is None:
                        from app.services.detection.tcclip_tensorrt_shadow import TcClipTensorRtShadow

                        self._shadow = TcClipTensorRtShadow(
                            engine_path=self.shadow_engine_path,
                            checkpoint_path=self.checkpoint_path,
                            prompts=prompts,
                            num_frames=self.num_frames,
                            sample_every=self.shadow_sample_every,
                            threshold=self.shadow_threshold,
                        )
                    if self._shadow is not None:
                        self._shadow.submit(tensors, result, prompts)
            except Exception as exc:
                self._shadow_error = f"{type(exc).__name__}: {exc}"
                logger.warning("TC-CLIP TensorRT shadow failed; PyTorch result retained: %s", exc)
        return result, prompts

    def _forward_tensor_batch(self, tensors: Sequence[torch.Tensor], num_classes: int) -> np.ndarray:
        batch = torch.stack(list(tensors)).to(self.device, non_blocking=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
        )
        with amp_context:
            output = self._model(batch)
            logits = aggregate_tcclip_logits(output["logits"].float(), self.prompts)
            probabilities = torch.softmax(logits, dim=1)
        if probabilities.ndim != 2 or probabilities.shape != (len(tensors), num_classes):
            raise RuntimeError(f"TC-CLIP returned an unexpected probability shape: {tuple(probabilities.shape)}")
        return probabilities.cpu().numpy().astype(np.float32)

    def get_info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "runtime_root": str(self.runtime_root),
            "classes": list(self.prompts.names),
            "prompt_classes": self.prompts.to_payload(),
            "prompt_count": self.prompts.prompt_count,
            "prompt_aggregation": "mean_logits",
            "normal_prompt": self.prompts.normal,
            "fight_prompt": self.prompts.fight,
            "num_frames": self.num_frames,
            "batch_size": self.batch_size,
            "context_length": TC_CLIP_CONTEXT_LENGTH,
            "stable_merge": self.stable_merge,
            "tensorrt_shadow": self._shadow.get_info() if self._shadow is not None else {
                "enabled": False,
                "configured": self.shadow_engine_path is not None,
                "last_error": self._shadow_error,
                "requires_stable_merge": bool(self.shadow_engine_path and not self.stable_merge),
            },
            "safe_batch_size": getattr(self, "_safe_batch_size", None),
            "oom_retries": getattr(self, "_oom_retries", 0),
            "last_inference": dict(getattr(self, "_last_inference", {})),
            "device": str(self.device),
            "loaded": self._model is not None,
        }


_SHARED_CLASSIFIERS: weakref.WeakValueDictionary[tuple[str, ...], TcClipActionClassifier] = (
    weakref.WeakValueDictionary()
)
_SHARED_CLASSIFIERS_LOCK = threading.Lock()


def update_shared_tcclip_prompts(prompts: TcClipPrompts) -> int:
    """Update every live shared classifier, including other detector subsystems."""
    with _SHARED_CLASSIFIERS_LOCK:
        classifiers = list(_SHARED_CLASSIFIERS.values())
    for classifier in classifiers:
        classifier.update_prompts(prompts)
    return len(classifiers)


def get_shared_tcclip_classifier(
    *,
    runtime_root: str | Path,
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    prompts: TcClipPrompts,
    device: torch.device,
    num_frames: int,
    batch_size: int,
    stable_merge: bool = False,
    shadow_engine_path: str | Path | None = None,
    shadow_sample_every: int = 20,
    shadow_threshold: float = 0.70,
) -> TcClipActionClassifier:
    """Share one heavy TC-CLIP model between stream detectors in a worker process."""
    key = (
        str(Path(runtime_root).expanduser().resolve()),
        str(Path(checkpoint_path).expanduser().resolve()),
        str(Path(cache_dir).expanduser().resolve()),
        str(device),
        str(int(num_frames)),
        str(int(batch_size)),
        str(bool(stable_merge)),
        str(shadow_engine_path or ""),
        str(int(shadow_sample_every)),
        str(float(shadow_threshold)),
    )
    with _SHARED_CLASSIFIERS_LOCK:
        classifier = _SHARED_CLASSIFIERS.get(key)
        if classifier is None:
            classifier = TcClipActionClassifier(
                runtime_root=runtime_root,
                checkpoint_path=checkpoint_path,
                cache_dir=cache_dir,
                prompts=prompts,
                device=device,
                num_frames=num_frames,
                batch_size=batch_size,
                stable_merge=stable_merge,
                shadow_engine_path=shadow_engine_path,
                shadow_sample_every=shadow_sample_every,
                shadow_threshold=shadow_threshold,
            )
            _SHARED_CLASSIFIERS[key] = classifier
        elif classifier.prompts != prompts:
            classifier.update_prompts(prompts)
        return classifier
