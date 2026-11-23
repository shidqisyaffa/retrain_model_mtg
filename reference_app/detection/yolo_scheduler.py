from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

YOLO_TRACK_CONFIDENCE = 0.20
YOLO_TRACK_IOU = 0.50


class YoloTrackProvider(Protocol):
    """Frame tracker supplied to detectors by an inference shard."""

    @property
    def device(self) -> str: ...

    @property
    def half(self) -> bool: ...

    def track(self, stream_id: str, frame: np.ndarray) -> object: ...

    def discard(self, stream_id: str) -> None: ...


@dataclass(slots=True)
class _BatchRequest[InputT, ResultT]:
    key: str
    value: InputT
    submitted_at: float
    event: threading.Event
    result: ResultT | None = None
    error: Exception | None = None
    cancelled: bool = False


class KeyedBatchWorker[InputT, ResultT]:
    """Bounded synchronous facade over one micro-batched worker thread."""

    def __init__(
        self,
        handler: Callable[[Sequence[tuple[str, InputT]]], Sequence[ResultT]],
        *,
        name: str,
        max_batch_size: int,
        batch_window_ms: float,
        max_pending: int,
        request_timeout_sec: float,
    ) -> None:
        if max_batch_size <= 0 or max_pending <= 0:
            raise ValueError("Batch size and pending capacity must be greater than zero")
        if batch_window_ms < 0 or request_timeout_sec <= 0:
            raise ValueError("Batch window cannot be negative and timeout must be positive")

        self._handler = handler
        self._max_batch_size = int(max_batch_size)
        self._batch_window_sec = float(batch_window_ms) / 1000.0
        self._max_pending = int(max_pending)
        self._request_timeout_sec = float(request_timeout_sec)
        self._condition = threading.Condition()
        self._pending: deque[_BatchRequest[InputT, ResultT]] = deque()
        self._pending_keys: set[str] = set()
        self._active: dict[str, _BatchRequest[InputT, ResultT]] = {}
        self._closed = False
        self._submitted = 0
        self._completed = 0
        self._batches = 0
        self._failures = 0
        self._timeouts = 0
        self._max_observed_batch = 0
        self._last_batch_ms = 0.0
        self._max_batch_ms = 0.0
        self._last_queue_wait_ms = 0.0
        self._max_queue_wait_ms = 0.0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, key: str, value: InputT) -> ResultT:
        deadline = time.monotonic() + self._request_timeout_sec
        request = _BatchRequest[InputT, ResultT](
            key=key,
            value=value,
            submitted_at=time.monotonic(),
            event=threading.Event(),
        )
        with self._condition:
            while len(self._pending) >= self._max_pending or key in self._pending_keys or key in self._active:
                if self._closed:
                    raise RuntimeError("YOLO batch worker is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timeouts += 1
                    raise TimeoutError(f"YOLO batch queue timed out for stream '{key}'")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("YOLO batch worker is closed")
            self._pending.append(request)
            self._pending_keys.add(key)
            self._submitted += 1
            self._condition.notify_all()

        remaining = max(0.0, deadline - time.monotonic())
        if not request.event.wait(remaining):
            request.cancelled = True
            with self._condition:
                self._timeouts += 1
            raise TimeoutError(f"YOLO batch inference timed out for stream '{key}'")
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("YOLO batch worker completed without a result")
        return request.result

    def discard(self, key: str) -> None:
        with self._condition:
            retained: deque[_BatchRequest[InputT, ResultT]] = deque()
            for request in self._pending:
                if request.key != key:
                    retained.append(request)
                    continue
                request.cancelled = True
                request.error = RuntimeError(f"YOLO request discarded for stream '{key}'")
                request.event.set()
            self._pending = retained
            self._pending_keys.discard(key)
            active = self._active.get(key)
            if active is not None and not active.cancelled:
                active.cancelled = True
                active.error = RuntimeError(f"YOLO request discarded for stream '{key}'")
                active.event.set()
            self._condition.notify_all()

    def stats(self) -> dict[str, int | float | bool]:
        with self._condition:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "batches": self._batches,
                "failures": self._failures,
                "timeouts": self._timeouts,
                "pending": len(self._pending),
                "active": len(self._active),
                "max_batch_size": self._max_batch_size,
                "max_observed_batch": self._max_observed_batch,
                "last_batch_ms": round(self._last_batch_ms, 1),
                "max_batch_ms": round(self._max_batch_ms, 1),
                "last_queue_wait_ms": round(self._last_queue_wait_ms, 1),
                "max_queue_wait_ms": round(self._max_queue_wait_ms, 1),
                "batch_window_ms": round(self._batch_window_sec * 1000.0, 3),
                "closed": self._closed,
                "thread_alive": self._thread.is_alive(),
            }

    def close(self, *, timeout: float = 2.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            while self._pending:
                request = self._pending.popleft()
                request.error = RuntimeError("YOLO batch worker closed")
                request.event.set()
            self._pending_keys.clear()
            for request in self._active.values():
                request.cancelled = True
                request.error = RuntimeError("YOLO batch worker closed")
                request.event.set()
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))

    def _take_batch(self) -> list[_BatchRequest[InputT, ResultT]]:
        batch: list[_BatchRequest[InputT, ResultT]] = []
        while self._pending and len(batch) < self._max_batch_size:
            request = self._pending.popleft()
            self._pending_keys.discard(request.key)
            if not request.cancelled:
                batch.append(request)
                self._active[request.key] = request
        return batch

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return

                deadline = time.monotonic() + self._batch_window_sec
                while len(self._pending) < self._max_batch_size and not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._closed:
                    return
                batch = self._take_batch()
                self._condition.notify_all()

            if not batch:
                continue
            batch_started = time.monotonic()
            queue_wait_ms = max((batch_started - request.submitted_at) * 1000.0 for request in batch)
            results: Sequence[ResultT] = ()
            error: Exception | None = None
            try:
                results = self._handler([(request.key, request.value) for request in batch])
                if len(results) != len(batch):
                    raise RuntimeError("YOLO batch result count does not match request count")
            except Exception as exc:
                error = exc

            batch_ms = (time.monotonic() - batch_started) * 1000.0
            with self._condition:
                self._batches += 1
                self._max_observed_batch = max(self._max_observed_batch, len(batch))
                self._last_batch_ms = batch_ms
                self._max_batch_ms = max(self._max_batch_ms, batch_ms)
                self._last_queue_wait_ms = queue_wait_ms
                self._max_queue_wait_ms = max(self._max_queue_wait_ms, queue_wait_ms)
                if error is not None:
                    self._failures += len(batch)
                for index, request in enumerate(batch):
                    self._active.pop(request.key, None)
                    self._completed += 1
                    if not request.cancelled:
                        request.error = error
                        if error is None:
                            request.result = results[index]
                        request.event.set()
                self._condition.notify_all()


class _SharedYoloRuntime:
    """One YOLO model with an isolated ByteTrack instance per stream."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        imgsz: int,
        half: bool,
        max_batch_size: int,
    ) -> None:
        import torch
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        from app.services.detection.yolo_backend import load_yolo

        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = str(torch.device(device))
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA was requested for YOLO but is unavailable; using CPU")
            resolved_device = "cpu"

        self.device = resolved_device
        self.half = bool(half and resolved_device.startswith("cuda"))
        self._imgsz = int(imgsz)
        self._checkpoint = model_path
        self._model, self.backend = load_yolo(
            model_path, device=resolved_device, imgsz=imgsz, max_batch=max_batch_size
        )
        tracker_path = check_yaml("bytetrack.yaml")
        self._tracker_args = IterableSimpleNamespace(**YAML.load(tracker_path))

        class ShardByteTracker(BYTETracker):
            @staticmethod
            def reset_id() -> None:
                # Ultralytics keeps its ID counter at process scope. Resetting
                # it while another shard is active can duplicate an existing
                # track ID, so new trackers keep the counter monotonic.
                return None

        self._tracker_type = ShardByteTracker
        self._trackers: dict[str, object] = {}
        self._generations: dict[str, int] = {}
        self._tracker_lock = threading.Lock()

    def discard(self, stream_id: str) -> None:
        with self._tracker_lock:
            self._generations[stream_id] = self._generations.get(stream_id, 0) + 1
            self._trackers.pop(stream_id, None)

    def generation(self, stream_id: str) -> int:
        with self._tracker_lock:
            return self._generations.get(stream_id, 0)

    def _predict_frames(self, frames: list[np.ndarray]) -> list[object]:
        return list(
            self._model.predict(
                source=frames,
                conf=YOLO_TRACK_CONFIDENCE,
                iou=YOLO_TRACK_IOU,
                imgsz=self._imgsz,
                device=(0 if self.device.startswith("cuda") else self.device),
                half=self.half,
                verbose=False,
            )
        )

    def _predict_frames_with_fallback(self, frames: list[np.ndarray]) -> list[object]:
        try:
            return self._predict_frames(frames)
        except Exception as exc:
            if self.backend != "tensorrt":
                raise
            logger.warning("YOLO TensorRT inference failed (%s); retrying same frames with PyTorch", exc)
            import torch
            from ultralytics import YOLO

            self._model = YOLO(self._checkpoint)
            self._model.to(torch.device(self.device))
            self.backend = "pytorch"
            return self._predict_frames(frames)

    def predict(
        self,
        items: Sequence[tuple[str, tuple[np.ndarray, int]]],
    ) -> list[object]:
        import torch

        frames = [value[0] for _, value in items]
        results = self._predict_frames_with_fallback(frames)
        if len(results) != len(items):
            raise RuntimeError("YOLO returned an unexpected batch size")

        tracked_results: list[object] = []
        for (stream_id, (_, generation)), result in zip(items, results, strict=True):
            with self._tracker_lock:
                if generation != self._generations.get(stream_id, 0):
                    tracked_results.append(result)
                    continue
                tracker = self._trackers.get(stream_id)
                if tracker is None:
                    tracker = self._tracker_type(self._tracker_args, frame_rate=30)
                    self._trackers[stream_id] = tracker
                boxes = result.boxes.cpu().numpy()
                tracks = tracker.update(boxes, result.orig_img)
            if len(tracks) > 0:
                indexes = tracks[:, -1].astype(int)
                result = result[indexes]
                result.update(boxes=torch.as_tensor(tracks[:, :-1]))
            tracked_results.append(result)
        return tracked_results


class SharedYoloScheduler:
    """Shared batched YOLO execution with per-camera ByteTrack state."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        imgsz: int = 640,
        half: bool = True,
        max_batch_size: int = 4,
        batch_window_ms: float = 8.0,
        max_pending: int = 8,
        request_timeout_sec: float = 10.0,
    ) -> None:
        self._runtime = _SharedYoloRuntime(
            model_path,
            device=device,
            imgsz=imgsz,
            half=half,
            max_batch_size=max_batch_size,
        )
        self._worker = KeyedBatchWorker[tuple[np.ndarray, int], object](
            self._runtime.predict,
            name=f"shared-yolo-{id(self):x}",
            max_batch_size=max_batch_size,
            batch_window_ms=batch_window_ms,
            max_pending=max_pending,
            request_timeout_sec=request_timeout_sec,
        )
        logger.info(
            "Shared YOLO scheduler online (device=%s, half=%s, batch=%d, window_ms=%.1f)",
            self.device,
            self.half,
            max_batch_size,
            batch_window_ms,
        )

    @property
    def device(self) -> str:
        return self._runtime.device

    @property
    def half(self) -> bool:
        return self._runtime.half

    @property
    def backend(self) -> str:
        return self._runtime.backend

    def track(self, stream_id: str, frame: np.ndarray) -> object:
        generation = self._runtime.generation(stream_id)
        return self._worker.submit(stream_id, (frame, generation))

    def discard(self, stream_id: str) -> None:
        self._worker.discard(stream_id)
        self._runtime.discard(stream_id)

    def stats(self) -> dict[str, int | float | bool | str]:
        return {
            **self._worker.stats(),
            "device": self.device,
            "half": self.half,
            "backend": self.backend,
        }

    def close(self, *, timeout: float = 2.0) -> None:
        self._worker.close(timeout=timeout)


class ProcessYoloScheduler:
    """One YOLO model and batch queue for all Scout and CCTV streams in this process."""

    def __init__(self, factory: Callable[[], SharedYoloScheduler] | None = None) -> None:
        self._factory = factory or self._create_scheduler
        self._lock = threading.RLock()
        self._scheduler: SharedYoloScheduler | None = None
        self._owners: set[object] = set()

    @staticmethod
    def _create_scheduler() -> SharedYoloScheduler:
        from app.core.config.config import settings

        batch_size = max(1, int(settings.CCTV_YOLO_BATCH_SIZE))
        return SharedYoloScheduler(
            settings.DETECTION_MODEL_PATH,
            device="auto",
            imgsz=settings.DETECTION_YOLO_IMGSZ,
            half=settings.DETECTION_YOLO_HALF,
            max_batch_size=batch_size,
            batch_window_ms=settings.CCTV_YOLO_BATCH_WINDOW_MS,
            max_pending=max(2, batch_size * 2),
            request_timeout_sec=settings.CCTV_YOLO_REQUEST_TIMEOUT_SEC,
        )

    def acquire(self, owner: object) -> SharedYoloScheduler:
        with self._lock:
            if self._scheduler is None:
                self._scheduler = self._factory()
            self._owners.add(owner)
            return self._scheduler

    def release(self, owner: object) -> None:
        with self._lock:
            if owner not in self._owners:
                return
            self._owners.remove(owner)
            if not self._owners and self._scheduler is not None:
                try:
                    self._scheduler.close(timeout=3.0)
                finally:
                    self._scheduler = None

    def stats(self) -> dict[str, int | float | bool | str]:
        with self._lock:
            if self._scheduler is None:
                return {"owners": 0, "closed": True}
            return {"owners": len(self._owners), **self._scheduler.stats()}


process_yolo_scheduler = ProcessYoloScheduler()
