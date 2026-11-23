"""Grid TC-CLIP detector — CCTV "High Performance" method for dense crowds.

Instead of YOLO + ByteTrack + per-person Dynamic ROI, this backend classifies
scene tiles: the frame is split into an ``N x N`` grid (with overlap), each tile
is sampled as a short clip and classified zero-shot by TC-CLIP. Results are
region-level detections (no person bbox), produced asynchronously through a
``KeyedLatestActionWorker`` so the camera capture loop is never blocked.

Preprocessing contract: tile crops are letterboxed to ``input_size`` (224) as
BGR arrays and passed through the shared ``TcClipActionClassifier`` — exactly
like ROI clips — so PyTorch and the prompt-dynamic TensorRT engine produce
comparable results. The classifier's own Resize(256)/CenterCrop(224) then
applies the same framing the ROI pipeline trains/infers with.

The grid classifier lives in a *separate* cache from the ROI classifier:
``update_shared_tcclip_prompts`` (ROI) must never overwrite grid prompts, and
vice versa.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import weakref
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.core.config.config import settings
from app.services.detection.action_queue import KeyedLatestActionWorker
from app.services.detection.detector import (
    BaseDetector,
    Detection,
    resolve_yolo_device,
)
from app.services.detection.tcclip_classifier import TcClipActionClassifier, TcClipPrompts
from app.services.detection.tcclip_engine_manager import TcClipEngineSelectionLog
from app.services.detection.tcclip_prompt_tokens import validate_tcclip_prompt_tokens

logger = logging.getLogger(__name__)

_GRID_CLASSIFIERS: weakref.WeakValueDictionary[tuple[str, ...], TcClipActionClassifier] = (
    weakref.WeakValueDictionary()
)
_GRID_CLASSIFIERS_LOCK = threading.Lock()

# Canvas padding colour used by the reference grid script (RGB 124,116,104).
_LETTERBOX_BGR = (104, 116, 124)

# Grid layout limits shared by the detector, pool, and service validation.
MAX_GRID_SIDE = 6
MAX_GRID_TILES = 36
# A grid clip must be continuous video, as for the ROI clips (dynamic_roi_tsm):
# after a gap in the source (the N8 hangar stream drops frames) the frames
# collected before it are dropped, never classified together with new ones.
# Never below 3 sample periods, so a low sample rate does not reset every time.
CLIP_MAX_GAP_SEC = 0.5

# Nama sel grid, mis. "R2C3" (1-based, sesuai label referensi).
_GRID_CELL_RE = re.compile(r"^R([1-6])C([1-6])$")


def normalize_grid_cells(cells: Any) -> set[str]:
    """Normalisasi nama sel nonaktif ("R1C2") — uppercase, valid, unik."""
    if not cells:
        return set()
    if isinstance(cells, str):
        cells = [cells]
    result: set[str] = set()
    for item in cells:
        name = str(item or "").strip().upper()
        if _GRID_CELL_RE.match(name):
            result.add(name)
    return result


def cell_in_layout(name: str, rows: int, cols: int) -> bool:
    """True bila nama sel berada di dalam layout rows x cols."""
    match = _GRID_CELL_RE.match(str(name or "").strip().upper())
    if not match:
        return False
    row, col = int(match.group(1)), int(match.group(2))
    return 1 <= row <= rows and 1 <= col <= cols


def mask_disabled_grid_classes(
    probabilities: np.ndarray, enabled_flags: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Per region: drop disabled classes and renormalise the rest.

    The same rule as the person/union detector
    (``DynamicRoiTcClipDetector._mask_disabled_probabilities``): renormalising
    softmax probabilities without some classes equals the softmax without them.
    Returns the masked rows and, per row, whether the disabled classes held
    (almost) all the probability. Such a region reports no action, never an
    alert, because promoting the small remainder would raise false alarms.
    """
    enabled = np.asarray(enabled_flags, dtype=bool)
    masked = np.where(enabled[None, :], probabilities, 0.0).astype(np.float32)
    remaining = masked.sum(axis=1, keepdims=True)
    has_enabled = remaining > 1e-9
    masked = np.divide(masked, remaining, out=np.zeros_like(masked), where=has_enabled)
    return masked, ~has_enabled[:, 0]


def get_grid_tcclip_classifier(
    *,
    runtime_root: str | Path,
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    prompts: TcClipPrompts,
    device: torch.device,
    num_frames: int,
    batch_size: int,
    stable_merge: bool = False,
    scope: str = "cctv",
) -> TcClipActionClassifier:
    """Share one grid TC-CLIP model per process and scope (separate from ROI prompts).

    ``scope`` keeps prompt sets that differ apart: the CCTV grid classes and the
    drone grid classes would otherwise overwrite each other on one classifier.
    """
    key = (
        str(scope),
        str(Path(runtime_root).expanduser().resolve()),
        str(Path(checkpoint_path).expanduser().resolve()),
        str(Path(cache_dir).expanduser().resolve()),
        str(device),
        str(int(num_frames)),
        str(int(batch_size)),
        str(bool(stable_merge)),
    )
    with _GRID_CLASSIFIERS_LOCK:
        classifier = _GRID_CLASSIFIERS.get(key)
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
            )
            _GRID_CLASSIFIERS[key] = classifier
        elif classifier.prompts != prompts:
            classifier.update_prompts(prompts)
        return classifier


def update_grid_tcclip_prompts(prompts: TcClipPrompts) -> int:
    """Apply new grid classes to every live grid classifier in this process."""
    with _GRID_CLASSIFIERS_LOCK:
        classifiers = list(_GRID_CLASSIFIERS.values())
    for classifier in classifiers:
        classifier.update_prompts(prompts)
    return len(classifiers)


@dataclass(frozen=True, slots=True)
class GridTile:
    """One grid region in working-frame pixel coordinates."""

    index: int
    name: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    rows: int
    cols: int


@dataclass(frozen=True, slots=True)
class GridJobGeometry:
    """Coordinates owned by one queued job, not mutable camera settings."""

    width: int
    height: int
    work_size: tuple[int, int]
    tiles: tuple[GridTile, ...]
    generation: int


@dataclass(frozen=True, slots=True)
class GridJobResult:
    detections: tuple[Detection, ...]
    completed_at: float
    geometry_generation: int
    prompt_generation: int


def make_grid_tiles(
    width: int,
    height: int,
    rows: int,
    cols: int,
    overlap: float,
) -> list[GridTile]:
    """Split a frame into an overlapping rows x cols grid (reference math)."""
    tiles: list[GridTile] = []
    for row in range(rows):
        for column in range(cols):
            left = max(0, int((column - overlap) * width / cols))
            top = max(0, int((row - overlap) * height / rows))
            right = min(width, int((column + 1 + overlap) * width / cols))
            bottom = min(height, int((row + 1 + overlap) * height / rows))
            if right <= left or bottom <= top:
                continue
            tiles.append(
                GridTile(
                    index=row * cols + column,
                    name=f"R{row + 1}C{column + 1}",
                    box=(left, top, right, bottom),
                    rows=rows,
                    cols=cols,
                )
            )
    return tiles


def _to_torch_device(friendly: str) -> torch.device:
    raw = resolve_yolo_device(friendly)
    if raw.isdecimal():
        return torch.device(f"cuda:{raw}")
    return torch.device(raw)


def letterbox_bgr(
    crop: np.ndarray,
    size: int,
    *,
    canvas_colour: tuple[int, int, int] = _LETTERBOX_BGR,
) -> np.ndarray:
    """Aspect-preserving resize with padding to ``size x size`` (BGR uint8)."""
    height, width = crop.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Empty grid crop")
    scale = min(size / width, size / height)
    new_size = (
        max(1, min(size, round(width * scale))),
        max(1, min(size, round(height * scale))),
    )
    resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), canvas_colour, dtype=np.uint8)
    offset_x = (size - new_size[0]) // 2
    offset_y = (size - new_size[1]) // 2
    canvas[offset_y : offset_y + new_size[1], offset_x : offset_x + new_size[0]] = resized
    return canvas


class TcClipGridDetector(BaseDetector):
    """Region-level TC-CLIP classifier for crowded CCTV scenes."""

    def __init__(
        self,
        *,
        stream_id: str,
        prompt_classes: Sequence[dict[str, object]] | None = None,
        grid_rows: int | None = None,
        grid_cols: int | None = None,
        tile_overlap: float | None = None,
        num_frames: int | None = None,
        sample_fps: float | None = None,
        inference_interval_sec: float | None = None,
        input_size: int | None = None,
        batch_size: int | None = None,
        result_ttl_sec: float | None = None,
        device: str = "auto",
        runtime_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        stable_merge: bool | None = None,
        engine_preference: str = "auto",
        action_recognition_enabled: bool = True,
        disabled_cells: Sequence[str] | None = None,
        classifier_scope: str = "cctv",
    ) -> None:
        self.stream_id = str(stream_id)
        self._classifier_scope = str(classifier_scope or "cctv")
        self._grid_rows = min(MAX_GRID_SIDE, max(1, int(grid_rows or settings.CCTV_GRID_ROWS)))
        self._grid_cols = min(MAX_GRID_SIDE, max(1, int(grid_cols or settings.CCTV_GRID_COLS)))
        # Sel grid yang dimatikan operator (tidak diklasifikasi & tak bisa alert).
        self._disabled_cells: set[str] = {
            name
            for name in normalize_grid_cells(disabled_cells)
            if cell_in_layout(name, self._grid_rows, self._grid_cols)
        }
        self._tile_overlap = float(
            settings.CCTV_GRID_TILE_OVERLAP if tile_overlap is None else tile_overlap
        )
        self._num_frames = max(2, int(num_frames or settings.CCTV_GRID_NUM_FRAMES))
        self._sample_fps = max(0.5, float(sample_fps or settings.CCTV_GRID_SAMPLE_FPS))
        self._interval = max(
            0.1,
            float(
                settings.CCTV_GRID_INFERENCE_INTERVAL_SEC
                if inference_interval_sec is None
                else inference_interval_sec
            ),
        )
        self._input_size = max(64, int(input_size or settings.CCTV_GRID_INPUT_SIZE))
        self._batch_size = max(1, int(batch_size or settings.CCTV_GRID_BATCH_SIZE))
        self._result_ttl = max(
            0.2,
            float(
                settings.CCTV_GRID_RESULT_TTL_SEC
                if result_ttl_sec is None
                else result_ttl_sec
            ),
        )
        self._device = _to_torch_device(device)
        self._runtime_root = Path(
            runtime_root or settings.DETECTION_TC_CLIP_RUNTIME_ROOT
        )
        self._checkpoint_path = Path(
            checkpoint_path or settings.DETECTION_TC_CLIP_CHECKPOINT_PATH
        )
        self._cache_dir = Path(cache_dir or settings.DETECTION_TC_CLIP_CACHE_DIR)
        self._stable_merge = bool(
            settings.DETECTION_TC_CLIP_STABLE_MERGE_ENABLED
            if stable_merge is None
            else stable_merge
        )
        self._engine_preference = str(engine_preference or "auto").strip().lower()
        self._action_recognition_enabled = bool(action_recognition_enabled)
        self._alert_threshold = float(settings.DETECTION_TC_CLIP_ALERTS_THRESHOLD)
        if not 0.0 < self._alert_threshold <= 1.0:
            raise ValueError("TC-CLIP alert threshold must be > 0 and <= 1")
        self._prompts = (
            TcClipPrompts(classes=list(prompt_classes)) if prompt_classes else None
        )

        self._classifier: TcClipActionClassifier | None = None
        self._runner: Any = None
        self._runner_error: str | None = None
        self._runner_attempt_generation: int | None = None
        self._engine_log = TcClipEngineSelectionLog(
            f"{self._classifier_scope.upper()}/Grid stream={self.stream_id}"
        )
        self._last_error: str | None = None

        self._lock = threading.RLock()
        self._width = 0
        self._height = 0
        self._work_size: tuple[int, int] = (0, 0)
        self._source_size = (0, 0)
        self._geometry_generation = 0
        self._tiles: list[GridTile] = []
        self._buffer: deque[np.ndarray] = deque(maxlen=self._num_frames)
        self._next_sample_at = 0.0
        self._last_sample_at = 0.0
        self._next_infer_at = 0.0
        self._inflight = False
        self._prompt_generation = 0
        self._inflight_generation = 0
        self._detections: list[Detection] = []
        self._detections_at = 0.0
        self._frames_detected = 0
        self._failures = 0
        self._worker: KeyedLatestActionWorker[int, Any, Any] = KeyedLatestActionWorker(
            lambda job: job(),
            name=f"tcclip-grid-{self.stream_id}",
        )

    # ── Geometry / sampling ─────────────────────────────────

    def _ensure_geometry(self) -> None:
        target_width = min(self._width, self._grid_cols * self._input_size)
        target_width = max(self._input_size, target_width)
        target_height = max(1, round(self._height * target_width / max(1, self._width)))
        if (
            (target_width, target_height) == self._work_size
            and (self._width, self._height) == self._source_size
        ):
            return
        self._invalidate_geometry()
        self._source_size = (self._width, self._height)
        self._work_size = (target_width, target_height)
        tiles = make_grid_tiles(
            target_width,
            target_height,
            self._grid_rows,
            self._grid_cols,
            self._tile_overlap,
        )
        if self._disabled_cells:
            tiles = [tile for tile in tiles if tile.name not in self._disabled_cells]
        self._tiles = tiles
        self._buffer.clear()

    def _invalidate_geometry(self) -> None:
        self._geometry_generation += 1
        self._work_size = (0, 0)
        self._tiles = []
        self._buffer.clear()
        self._detections = []
        self._detections_at = 0.0
        # Keep _inflight until its completion is drained. A new job must not
        # inherit the old job's identity or overwrite its generation check.

    def set_disabled_cells(self, cells: Any) -> bool:
        """Terapkan sel grid yang dimatikan (live, tanpa reload model).

        Geometry di-rebuild sehingga tile nonaktif tidak diklasifikasi sama
        sekali — hemat compute dan tidak bisa memicu alert.
        """
        normalized = normalize_grid_cells(cells)
        with self._lock:
            normalized = {
                name for name in normalized
                if cell_in_layout(name, self._grid_rows, self._grid_cols)
            }
            if normalized == self._disabled_cells:
                return False
            self._disabled_cells = normalized
            self._invalidate_geometry()
            return True

    def reset_clip(self) -> None:
        """Drop the frames collected so far and the last result.

        A clip must show one still view; drones call this while moving or
        right after a gimbal/zoom/lens command, so no half-moved clip is classified.
        """
        with self._lock:
            self._invalidate_geometry()

    def disabled_cells(self) -> list[str]:
        with self._lock:
            return sorted(self._disabled_cells)

    def set_grid_layout(self, rows: int, cols: int) -> bool:
        """Apply a new rows x cols layout live (no model reload)."""
        rows_key = min(MAX_GRID_SIDE, max(1, int(rows)))
        cols_key = min(MAX_GRID_SIDE, max(1, int(cols)))
        with self._lock:
            changed = (rows_key, cols_key) != (self._grid_rows, self._grid_cols)
            if not changed:
                return False
            self._grid_rows = rows_key
            self._grid_cols = cols_key
            # Nama sel di luar layout baru tidak lagi relevan.
            self._disabled_cells = {
                name
                for name in self._disabled_cells
                if cell_in_layout(name, rows_key, cols_key)
            }
            self._invalidate_geometry()
            return changed

    def grid_layout(self) -> dict[str, int]:
        with self._lock:
            return {
                "rows": self._grid_rows,
                "cols": self._grid_cols,
                "tiles": len(self._tiles),
            }

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        work_width, work_height = self._work_size
        if frame.shape[1] == work_width and frame.shape[0] == work_height:
            return frame
        return cv2.resize(
            frame,
            (work_width, work_height),
            interpolation=cv2.INTER_AREA,
        )

    # ── Models ──────────────────────────────────────────────

    def _ensure_classifier(self) -> TcClipActionClassifier | None:
        if self._classifier is not None:
            return self._classifier
        if self._prompts is None:
            self._last_error = "Kelas grid belum dikonfigurasi"
            return None
        try:
            self._classifier = get_grid_tcclip_classifier(
                runtime_root=self._runtime_root,
                checkpoint_path=self._checkpoint_path,
                cache_dir=self._cache_dir,
                prompts=self._prompts,
                device=self._device,
                num_frames=self._num_frames,
                batch_size=self._batch_size,
                stable_merge=self._stable_merge,
                scope=self._classifier_scope,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to no detections
            self._last_error = f"Grid TC-CLIP load failed: {type(exc).__name__}: {exc}"
            logger.warning("Grid TC-CLIP load failed stream=%s: %s", self.stream_id, exc)
            return None
        return self._classifier

    def set_prompt_classes(self, prompt_classes: Sequence[dict[str, object]]) -> None:
        prompts = TcClipPrompts(classes=list(prompt_classes))
        validate_tcclip_prompt_tokens(prompts.to_payload(), runtime_root=self._runtime_root)
        with self._lock:
            if self._prompts is not None and prompts == self._prompts:
                return
            if self._classifier is not None:
                self._classifier.update_prompts(prompts)
            self._prompts = prompts
            self._prompt_generation += 1
            self._detections = []
            self._detections_at = 0.0
            self._runner = None
            self._runner_error = None
            self._runner_attempt_generation = None

    def set_tcclip_inference_engine(self, engine: str) -> None:
        key = str(engine or "auto").strip().lower()
        if key not in {"auto", "pytorch", "tensorrt"}:
            return
        with self._lock:
            if key != self._engine_preference:
                self._runner_attempt_generation = None
            self._engine_preference = key
            if key == "pytorch":
                self._runner = None
                self._runner_error = None
                return
        self._ensure_runner()

    def _ensure_runner(self) -> Any:
        runner = self._select_runner()
        if self._prompts is not None:
            if runner is not None:
                reason = "compatible CUDA TensorRT engine"
            elif self._engine_preference == "pytorch":
                reason = "explicit legacy PyTorch override"
            else:
                reason = "TensorRT not ready"
            self._engine_log.report(
                "tensorrt" if runner is not None else "pytorch",
                device=self._device,
                reason=self._runner_error or reason,
            )
        return runner

    def _select_runner(self) -> Any:
        if self._engine_preference == "pytorch" or self._prompts is None:
            return None
        if self._runner is not None:
            return self._runner
        if self._device.type != "cuda":
            self._runner_error = "CUDA unavailable"
            return None
        if not self._stable_merge:
            self._runner_error = (
                "TC-CLIP TensorRT engine requires detection.tcclip_stable_merge_enabled: true"
            )
            return None
        try:
            from app.services.detection.tcclip_engine_manager import (
                tcclip_engine_manager,
            )
            from app.services.detection.tcclip_tensorrt_shadow import (
                TensorRtRunner,
                runner_accepts_prompts,
            )
        except Exception as exc:  # noqa: BLE001
            self._runner_error = f"{type(exc).__name__}: {exc}"
            return None
        build = tcclip_engine_manager.status(self._prompts)
        generation = int(build["generation"])
        # Do not reload an unavailable or failed engine for every grid job.
        # A completed background build changes generation and enables a retry.
        if self._runner_attempt_generation == generation:
            return None
        self._runner_attempt_generation = generation
        errors: list[str] = []
        for path in tcclip_engine_manager.candidate_paths():
            try:
                runner = TensorRtRunner(
                    path,
                    checkpoint_path=self._checkpoint_path,
                    prompts=self._prompts,
                    num_frames=self._num_frames,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path.name}: {exc}")
                continue
            if runner_accepts_prompts(runner, self._prompts):
                self._runner = runner
                self._runner_error = None
                return runner
            errors.append(f"{path.name}: prompts tidak cocok")
        if build["state"] == "failed":
            self._runner_error = str(build["error"])
            return None
        tcclip_engine_manager.request(self._prompts, force=build["state"] == "ready")
        self._runner_error = "TensorRT engine sedang dibangun; sementara PyTorch: " + "; ".join(errors)
        return None

    def engine_status(self) -> dict[str, Any]:
        from app.services.detection.tcclip_engine_manager import tcclip_engine_manager

        prompts = self._prompts
        build = (
            tcclip_engine_manager.status(prompts)
            if prompts is not None
            else {"state": "idle", "error": None, "generation": 0}
        )
        return {
            "preference": self._engine_preference,
            "active": "tensorrt" if self._runner is not None else "pytorch",
            "error": self._runner_error,
            "build": build,
        }

    # ── BaseDetector ────────────────────────────────────────

    def set_feature_flags(
        self,
        *,
        action_recognition_enabled: bool = True,
        object_detection_enabled: bool = True,
    ) -> None:
        del object_detection_enabled
        with self._lock:
            enabled = bool(action_recognition_enabled)
            if enabled != self._action_recognition_enabled:
                self._action_recognition_enabled = enabled
                self._invalidate_geometry()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detect_with_context(frame)

    def detect_with_context(
        self,
        frame: np.ndarray,
        *,
        frame_id: int | None = None,
        captured_at: float | None = None,
        include_actions: bool = False,
    ) -> list[Detection]:
        del include_actions
        if frame is None or frame.size == 0 or not self._action_recognition_enabled:
            return []
        if self._worker is None:
            return []
        now = time.monotonic()
        with self._lock:
            self._width = int(frame.shape[1])
            self._height = int(frame.shape[0])
            self._ensure_geometry()

            if now >= self._next_sample_at:
                max_gap = max(CLIP_MAX_GAP_SEC, 3.0 / self._sample_fps)
                if self._buffer and self._last_sample_at and now - self._last_sample_at > max_gap:
                    # Video gap: the frames so far no longer join up with this one.
                    self._buffer.clear()
                self._buffer.append(self._downscale(frame))
                self._last_sample_at = now
                self._next_sample_at = now + 1.0 / self._sample_fps

            if (
                len(self._buffer) >= self._num_frames
                and now >= self._next_infer_at
                and not self._inflight
            ):
                frames = list(self._buffer)
                self._next_infer_at = now + self._interval
                self._inflight = True
                self._inflight_generation = self._prompt_generation
                geometry = self._job_geometry()
                prompt_generation = self._prompt_generation
                task_id = self._worker.submit(
                    0,
                    lambda: self._run_grid_job(
                        frames,
                        int(frame_id or 0),
                        float(captured_at or time.time()),
                        geometry,
                        prompt_generation,
                    ),
                )
                if task_id is None:
                    self._inflight = False

            result = self._worker.poll_latest(0)
            if result is not None:
                self._inflight = False
                if result.error is not None:
                    self._failures += 1
                    self._last_error = f"{type(result.error).__name__}: {result.error}"
                elif (
                    isinstance(result.value, GridJobResult)
                    and result.value.geometry_generation == self._geometry_generation
                    and result.value.prompt_generation == self._prompt_generation
                ):
                    self._detections = list(result.value.detections)
                    self._detections_at = result.value.completed_at
                    self._frames_detected += 1
                    self._last_error = None

            if self._detections and (now - self._detections_at) > self._result_ttl:
                self._detections = []
            return list(self._detections)

    def _job_geometry(self) -> GridJobGeometry:
        return GridJobGeometry(
            self._width, self._height, self._work_size,
            tuple(self._tiles), self._geometry_generation,
        )

    def _run_grid_job(
        self, frames: list[np.ndarray], source_frame_id: int, captured_at: float,
        geometry: GridJobGeometry, prompt_generation: int,
    ) -> GridJobResult:
        detections, completed_at = self._run_grid(
            frames, source_frame_id, captured_at, geometry=geometry,
        )
        return GridJobResult(
            tuple(detections), completed_at, geometry.generation, prompt_generation,
        )

    def _run_grid(
        self,
        frames: list[np.ndarray],
        source_frame_id: int,
        captured_at: float,
        *,
        geometry: GridJobGeometry | None = None,
    ) -> tuple[list[Detection], float]:
        if geometry is None:
            with self._lock:
                geometry = self._job_geometry()
        classifier = self._ensure_classifier()
        if classifier is None or not geometry.tiles:
            return [], time.monotonic()
        clips: list[list[np.ndarray]] = []
        selected: list[GridTile] = []
        for tile in geometry.tiles:
            clip = self._tile_clip(frames, tile)
            if clip is None:
                continue
            clips.append(clip)
            selected.append(tile)
        if not clips:
            return [], time.monotonic()

        runner = self._ensure_runner()
        try:
            probabilities, prompts = classifier.predict_with_prompts(
                clips,
                batch_size=self._batch_size,
                tensorrt_runner=runner,
            )
        except Exception as exc:
            if runner is None:
                raise
            error = f"TensorRT inference failed: {type(exc).__name__}: {exc}"
            with self._lock:
                if self._runner is runner:
                    self._runner = None
                    self._runner_error = error
            logger.warning("Grid TC-CLIP TensorRT failed stream=%s; retrying with PyTorch: %s", self.stream_id, exc)
            self._engine_log.report("pytorch", device=self._device, reason=error)
            probabilities, prompts = classifier.predict_with_prompts(
                clips,
                batch_size=self._batch_size,
                tensorrt_runner=None,
            )
        classes = list(prompts.classes)
        if not classes or probabilities.shape[1] != len(classes):
            return [], time.monotonic()
        # Disabled classes stay model candidates (same text, engine and shared
        # classifier) but never win a region.
        only_disabled = np.zeros(len(selected), dtype=bool)
        if not all(prompts.enabled_flags):
            probabilities, only_disabled = mask_disabled_grid_classes(
                probabilities, prompts.enabled_flags
            )
        baseline = next((item for item in classes if item.name == prompts.baseline_name), classes[0])

        work_width, work_height = geometry.work_size
        scale_x = geometry.width / max(1, work_width)
        scale_y = geometry.height / max(1, work_height)
        detections: list[Detection] = []
        for tile, row, no_enabled in zip(selected, probabilities, only_disabled, strict=True):
            if no_enabled:
                prompt_class, confidence = baseline, 0.0
            else:
                top_index = int(np.argmax(row))
                prompt_class = classes[top_index]
                confidence = float(row[top_index])
            left, top, right, bottom = tile.box
            x1 = max(0, min(geometry.width, round(left * scale_x)))
            y1 = max(0, min(geometry.height, round(top * scale_y)))
            x2 = max(0, min(geometry.width, round(right * scale_x)))
            y2 = max(0, min(geometry.height, round(bottom * scale_y)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    confidence=confidence,
                    label=f"{tile.name} {prompt_class.name}",
                    action=prompt_class.name,
                    # Keep the winning label/score visible, but gate alarms by
                    # the same configured confidence threshold as ROI/Scout.
                    is_alert=bool(prompt_class.is_alert and confidence >= self._alert_threshold),
                    threat_level=prompt_class.threat_level,
                    roi_type="region",
                    # Immutable tile layout, never a later camera setting.
                    # The inference bbox/crop above deliberately retains overlap.
                    grid_cell={
                        "row": tile.index // tile.cols + 1,
                        "col": tile.index % tile.cols + 1,
                        "rows": tile.rows,
                        "cols": tile.cols,
                    },
                )
            )
        del source_frame_id, captured_at
        return detections, time.monotonic()

    def _tile_clip(
        self,
        frames: Sequence[np.ndarray],
        tile: GridTile,
    ) -> list[np.ndarray] | None:
        left, top, right, bottom = tile.box
        clip: list[np.ndarray] = []
        for frame in frames:
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                return None
            clip.append(letterbox_bgr(crop, self._input_size))
        return clip

    def get_info(self) -> dict:
        return {
            "backend": "tcclip_grid",
            "mode": "grid_zero_shot",
            "stream_id": self.stream_id,
            "grid_rows": self._grid_rows,
            "grid_cols": self._grid_cols,
            "tiles": [tile.name for tile in self._tiles],
            "disabled_cells": sorted(self._disabled_cells),
            "enabled_tiles": len(self._tiles),
            "tile_overlap": self._tile_overlap,
            "num_frames": self._num_frames,
            "sample_fps": self._sample_fps,
            "inference_interval_sec": self._interval,
            "input_size": self._input_size,
            "batch_size": self._batch_size,
            "alert_threshold": self._alert_threshold,
            "device": str(self._device),
            "prompt_classes": self._prompts.to_payload() if self._prompts else [],
            "tcclip_engine": self.engine_status(),
            "frames_detected": self._frames_detected,
            "failures": self._failures,
            "inflight": self._inflight,
            "worker": self._worker.stats(),
            "last_error": self._last_error,
        }

    def close(self) -> None:
        worker, self._worker = self._worker, None  # type: ignore[assignment]
        if worker is not None:
            worker.close(timeout=1.0)
