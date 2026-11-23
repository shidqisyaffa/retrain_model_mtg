from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config.config import settings
from app.services.detection.action_queue import KeyedLatestActionWorker
from app.services.detection.detector import ActionDetectionBatch, Detection
from app.services.detection.dynamic_roi_tsm import (
    PREDICT_EVERY_SAMPLES,
    ActionPrediction,
    BaseROIState,
    DynamicRoiTsmDetector,
    GroupROIState,
    IndividualROIState,
    PersonTrack,
)
from app.services.detection.tcclip_batch_queue import KeyedLatestBatchActionWorker
from app.services.detection.tcclip_classifier import (
    TcClipActionClassifier,
    TcClipPrompts,
    get_shared_tcclip_classifier,
)
from app.services.detection.tcclip_engine_manager import TcClipEngineSelectionLog
from app.services.detection.yolo_scheduler import YoloTrackProvider

logger = logging.getLogger(__name__)
TC_CLIP_ERROR_LOG_INTERVAL_SEC = 10.0


@dataclass(frozen=True, slots=True)
class _TcClipActionTask:
    source_frame_id: int
    source_captured_at: float
    submitted_at: float
    generation: int
    states: tuple[BaseROIState, ...]
    clips: tuple[tuple[np.ndarray, ...], ...]


@dataclass(frozen=True, slots=True)
class _TcClipActionOutput:
    task: _TcClipActionTask
    probabilities: np.ndarray
    prompts: TcClipPrompts


@dataclass(frozen=True, slots=True)
class TcClipRoiBatchJob:
    classifier: TcClipActionClassifier
    task: _TcClipActionTask


def run_tcclip_roi_batch(
    jobs: Sequence[TcClipRoiBatchJob], *, model_batch_size: int | None = None, tensorrt_runner: object | None = None
) -> list[_TcClipActionOutput]:
    """Run ROI clips from all CCTV cameras in shared model forward passes."""
    outputs: list[_TcClipActionOutput | None] = [None] * len(jobs)
    by_classifier: dict[int, list[tuple[int, TcClipRoiBatchJob]]] = {}
    for index, job in enumerate(jobs):
        by_classifier.setdefault(id(job.classifier), []).append((index, job))
    for members in by_classifier.values():
        classifier = members[0][1].classifier
        clips = [job.task.clips[0] for _, job in members]
        shadow = isinstance(classifier, TcClipActionClassifier)
        if shadow and tensorrt_runner is not None:
            probabilities, prompts = classifier.predict_with_prompts(
                clips, batch_size=model_batch_size, tensorrt_runner=tensorrt_runner
            )
        elif model_batch_size is None:
            if shadow:
                probabilities, prompts = classifier.predict_with_prompts(clips, shadow=True)
            else:
                probabilities, prompts = classifier.predict_with_prompts(clips)
        elif shadow:
            probabilities, prompts = classifier.predict_with_prompts(
                clips, batch_size=model_batch_size, shadow=True
            )
        else:
            probabilities, prompts = classifier.predict_with_prompts(clips, batch_size=model_batch_size)
        if len(probabilities) != len(members):
            raise RuntimeError("TC-CLIP returned an unexpected ROI count")
        for row, (index, job) in enumerate(members):
            outputs[index] = _TcClipActionOutput(
                task=job.task,
                probabilities=probabilities[row : row + 1],
                prompts=prompts,
            )
    if any(output is None for output in outputs):
        raise RuntimeError("TC-CLIP did not return every ROI result")
    return [output for output in outputs if output is not None]


class DynamicRoiTcClipDetector(DynamicRoiTsmDetector):
    """YOLO + Dynamic ROI detector using local TC-CLIP action classification."""

    def __init__(
        self,
        *,
        yolo_model_path: str,
        tcclip_runtime_root: str,
        tcclip_checkpoint_path: str,
        tcclip_cache_dir: str,
        normal_prompt: str,
        fight_prompt: str,
        prompt_classes: Sequence[dict[str, object]] | None = None,
        fight_threshold: float = 0.70,
        num_frames: int = 16,
        temporal_stride: int = 2,
        target_fps: float = 10.0,
        batch_size: int = 1,
        inference_interval_sec: float = 0.8,
        max_rois_per_job: int = 2,
        max_result_age_sec: float = 2.0,
        action_ttl_sec: float = 2.0,
        conf_person: float = 0.30,
        conf_weapon: float = 0.25,
        device: str = "auto",
        contrast_alpha: float = 1.0,
        contrast_beta: float = 0.0,
        yolo_imgsz: int = 640,
        yolo_half: bool = True,
        action_recognition_enabled: bool = True,
        object_detection_enabled: bool = True,
        action_scheduler: KeyedLatestActionWorker[str, Callable[[], object], object]
        | KeyedLatestBatchActionWorker[tuple[str, str, int], str, TcClipRoiBatchJob, _TcClipActionOutput]
        | None = None,
        yolo_scheduler: YoloTrackProvider | None = None,
        yolo_stream_id: str | None = None,
        alerts_threshold: float | None = None,
    ) -> None:
        self._tcclip_runtime_root = str(Path(tcclip_runtime_root).expanduser())
        self._tcclip_checkpoint_path = str(Path(tcclip_checkpoint_path).expanduser())
        self._tcclip_cache_dir = str(Path(tcclip_cache_dir).expanduser())
        self._tcclip_prompts = (
            TcClipPrompts(classes=prompt_classes)
            if prompt_classes is not None
            else TcClipPrompts(normal_prompt, fight_prompt)
        )
        # fight_threshold remains a legacy keyword; an explicit new name wins.
        self._tcclip_alerts_threshold = float(alerts_threshold if alerts_threshold is not None else fight_threshold)
        self._tcclip_num_frames = int(num_frames)
        self._tcclip_temporal_stride = int(temporal_stride)
        self._tcclip_target_fps = float(target_fps)
        self._tcclip_batch_size = int(batch_size)
        self._inference_interval_sec = float(inference_interval_sec)
        self._max_rois_per_job = int(max_rois_per_job)
        self._max_result_age_sec = float(max_result_age_sec)
        self._action_ttl_sec = float(action_ttl_sec)
        self._tcclip_classifier: TcClipActionClassifier | None = None
        self._tcclip_auto_runner = None
        self._tcclip_auto_error: str | None = None
        self._tcclip_auto_generation_seen = -1
        # Automatic by default; explicit setters remain for legacy diagnostics.
        self._tcclip_engine_preference = "auto"
        self._tcclip_engine_log = TcClipEngineSelectionLog(f"ROI stream={yolo_stream_id or 'direct'}")
        self._prediction_prompts = self._tcclip_prompts
        self._tcclip_last_error: str | None = None
        self._tcclip_last_error_log_at = 0.0
        self._action_worker = action_scheduler
        self._owns_action_worker = action_scheduler is None
        self._action_queue_key = f"tcclip-{yolo_stream_id}" if yolo_stream_id else f"tcclip-{id(self):x}"
        self._roi_batch_mode = isinstance(action_scheduler, KeyedLatestBatchActionWorker)
        self._action_roi_keys: set[tuple[str, str, int]] = set()
        self._action_roi_keys_lock = threading.RLock()
        self._completed_action_updates: list[ActionDetectionBatch] = []
        self._action_generation = 0
        self._current_frame_id = 0
        self._stale_action_results = 0
        self._rate_limited_submissions = 0
        self._admission_rejections = 0
        self._roi_last_submitted_at: dict[tuple[str, int], float] = {}
        self._next_action_submit_at = 0.0
        self._roi_cursor = 0
        self._priority_roi_cursor = 0
        self._current_capture_at = time.monotonic()
        self._action_last_applied: dict[tuple[str, int], float] = {}

        if not 0.0 < self._tcclip_alerts_threshold <= 1.0:
            raise ValueError("TC-CLIP alerts_threshold must be > 0 and <= 1")
        if self._tcclip_temporal_stride < 1:
            raise ValueError("TC-CLIP temporal_stride must be at least 1")
        if self._tcclip_target_fps <= 0.0:
            raise ValueError("TC-CLIP target_fps must be positive")
        if self._inference_interval_sec < 0.0:
            raise ValueError("TC-CLIP inference_interval_sec must not be negative")
        if self._max_rois_per_job < 1:
            raise ValueError("TC-CLIP max_rois_per_job must be at least 1")
        if self._max_result_age_sec <= 0.0 or self._action_ttl_sec <= 0.0:
            raise ValueError("TC-CLIP result age and action TTL must be positive")

        super().__init__(
            yolo_model_path=yolo_model_path,
            tsm_model_path=self._tcclip_checkpoint_path,
            tsm_classes=list(self._tcclip_prompts.names),
            alert_classes=list(self._tcclip_prompts.alert_names),
            num_segments=self._tcclip_num_frames,
            inference_interval=1,
            batch_size=self._tcclip_batch_size,
            conf_person=conf_person,
            conf_weapon=conf_weapon,
            device=device,
            contrast_alpha=contrast_alpha,
            contrast_beta=contrast_beta,
            yolo_imgsz=yolo_imgsz,
            yolo_half=yolo_half,
            tsm_compile=False,
            action_recognition_enabled=action_recognition_enabled,
            object_detection_enabled=object_detection_enabled,
            yolo_scheduler=yolo_scheduler,
            yolo_stream_id=yolo_stream_id,
        )
        if self._tcclip_classifier is not None and self._action_worker is None:
            self._action_worker = KeyedLatestActionWorker(
                lambda job: job(),
                name=f"tcclip-action-{id(self):x}",
            )

    def _load_models(self) -> None:
        self._load_yolo()
        assert self._device is not None
        try:
            self._tcclip_classifier = get_shared_tcclip_classifier(
                runtime_root=self._tcclip_runtime_root,
                checkpoint_path=self._tcclip_checkpoint_path,
                cache_dir=self._tcclip_cache_dir,
                prompts=self._tcclip_prompts,
                device=self._device,
                num_frames=self._tcclip_num_frames,
                batch_size=self._tcclip_batch_size,
                stable_merge=settings.DETECTION_TC_CLIP_STABLE_MERGE_ENABLED,
                shadow_engine_path=(
                    settings.CCTV_TC_CLIP_TENSORRT_ENGINE_PATH
                    if settings.CCTV_TC_CLIP_TENSORRT_SHADOW_ENABLED else None
                ),
                shadow_sample_every=settings.CCTV_TC_CLIP_TENSORRT_SHADOW_SAMPLE_EVERY,
                shadow_threshold=self._tcclip_alerts_threshold,
            )
            self._tcclip_last_error = None
        except Exception as exc:
            self._tcclip_classifier = None
            self._record_tcclip_error(exc, context="model load")
        self._select_direct_tcclip_engine()
        self._tsm = None
        self._transform = None
        self._tsm_classes = list(self._tcclip_prompts.names)
        self.set_alert_classes_with_levels({
            item.name: item.threat_level for item in self._tcclip_prompts.classes if item.is_alert
        })
        self._num_segments = self._tcclip_num_frames
        self._temporal_stride = self._tcclip_temporal_stride
        self._target_fps = self._tcclip_target_fps
        self._clip_span = (self._num_segments - 1) * self._temporal_stride + 1
        self._tsm_compile = False

    def set_tcclip_inference_engine(self, engine: str) -> None:
        """Apply the shared TC-CLIP engine preference (auto/pytorch/tensorrt).

        "pytorch" disables the direct TensorRT runner; the other values re-run
        the candidate selection (building the dynamic engine in the background
        when it is missing).
        """
        key = str(engine or "auto").strip().lower()
        if key not in {"auto", "pytorch", "tensorrt"}:
            return
        self._tcclip_engine_preference = key
        if key == "pytorch":
            self._tcclip_auto_runner = None
            self._tcclip_auto_error = None
            self._report_direct_tcclip_engine()
            return
        self._select_direct_tcclip_engine()

    def engine_status(self) -> dict[str, object]:
        """Compact engine info for the Scout AI snapshot."""
        return {
            "preference": self._tcclip_engine_preference,
            "active": "tensorrt" if getattr(self, "_tcclip_auto_runner", None) is not None else "pytorch",
            "build": self._tcclip_engine_build_status(),
        }

    def _select_direct_tcclip_engine(self) -> None:
        self._select_direct_tcclip_engine_impl()
        self._report_direct_tcclip_engine()

    def _report_direct_tcclip_engine(self) -> None:
        # CCTV batch mode is reported by its shared pool, not each ROI detector.
        if getattr(self, "_roi_batch_mode", False) or getattr(self, "_tcclip_classifier", None) is None:
            return
        reporter = getattr(self, "_tcclip_engine_log", None)
        if reporter is None:
            reporter = self._tcclip_engine_log = TcClipEngineSelectionLog("ROI")
        device = self._device
        runner = self._tcclip_auto_runner
        if runner is not None:
            reason = "compatible CUDA TensorRT engine"
        elif getattr(self, "_tcclip_engine_preference", "auto") == "pytorch":
            reason = "explicit legacy PyTorch override"
        else:
            reason = "CUDA unavailable" if device.type != "cuda" else "automatic selection"
        reason = self._tcclip_auto_error or reason
        reporter.report("tensorrt" if runner is not None else "pytorch", device=device, reason=reason)

    def _select_direct_tcclip_engine_impl(self) -> None:
        from app.services.detection.tcclip_engine_manager import tcclip_engine_manager
        from app.services.detection.tcclip_tensorrt_shadow import TensorRtRunner, runner_accepts_prompts

        if getattr(self, "_tcclip_engine_preference", "auto") == "pytorch":
            self._tcclip_auto_runner = None
            return
        current = getattr(self, "_tcclip_auto_runner", None)
        if current is not None and runner_accepts_prompts(current, self._tcclip_prompts):
            return
        self._tcclip_auto_runner = None
        self._tcclip_auto_error = None
        if getattr(self, "_roi_batch_mode", False) or self._tcclip_classifier is None or self._device.type != "cuda":
            return
        if not getattr(self._tcclip_classifier, "stable_merge", False):
            self._tcclip_auto_error = "TC-CLIP TensorRT engine requires detection.tcclip_stable_merge_enabled: true"
            return
        build = tcclip_engine_manager.status(self._tcclip_prompts)
        self._tcclip_auto_generation_seen = int(build["generation"])
        errors = []
        for path in tcclip_engine_manager.candidate_paths():
            try:
                self._tcclip_auto_runner = TensorRtRunner(
                    path,
                    checkpoint_path=Path(self._tcclip_checkpoint_path),
                    prompts=self._tcclip_prompts,
                    num_frames=self._tcclip_num_frames,
                )
                return
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        if build["state"] == "failed":
            self._tcclip_auto_error = str(build["error"])
            return
        tcclip_engine_manager.request(self._tcclip_prompts, force=build["state"] == "ready")
        self._tcclip_auto_error = "TensorRT engine sedang dibangun; sementara PyTorch: " + "; ".join(errors)

    def _predict_direct_tcclip(self, clips):
        classifier = self._tcclip_classifier
        if classifier is None:
            raise RuntimeError("TC-CLIP classifier is not loaded")
        if getattr(self, "_tcclip_auto_runner", None) is None and getattr(getattr(self, "_device", None), "type", None) == "cuda":
            from app.services.detection.tcclip_engine_manager import tcclip_engine_manager

            build = tcclip_engine_manager.status(self._tcclip_prompts)
            if int(build["generation"]) != getattr(self, "_tcclip_auto_generation_seen", -1):
                self._select_direct_tcclip_engine()
        runner = getattr(self, "_tcclip_auto_runner", None)
        if runner is not None:
            try:
                return classifier.predict_with_prompts(clips, tensorrt_runner=runner)
            except Exception as exc:
                self._tcclip_auto_runner = None
                self._tcclip_auto_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Scout TC-CLIP TensorRT failed; retrying with PyTorch: %s", exc)
                self._report_direct_tcclip_engine()
        return classifier.predict_with_prompts(clips)

    def _action_model_ready(self) -> bool:
        return self._tcclip_classifier is not None and self._action_worker is not None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Backward-compatible direct API with asynchronously cached labels."""
        self._current_capture_at = time.monotonic()
        self._consume_action_result()
        self._expire_action_labels()
        return super().detect(frame)

    def detect_with_context(
        self,
        frame: np.ndarray,
        *,
        frame_id: int | None = None,
        captured_at: float | None = None,
        include_actions: bool = False,
    ) -> list[Detection]:
        self._source_frame_id_override = frame_id
        self._current_capture_at = time.monotonic() if captured_at is None else float(captured_at)
        try:
            self._consume_action_result()
            self._expire_action_labels()
            detections = super().detect(frame)
        finally:
            self._source_frame_id_override = None
        if include_actions:
            return detections
        return [self._position_only_detection(detection) for detection in detections]

    def _select_current_individual_ids(self, sorted_people: Sequence[PersonTrack]) -> set[int]:
        # Every person keeps a temporal buffer; the shared CCTV scheduler
        # takes a fair turn per ROI and limits each GPU batch.
        return {detection.track_id for detection in sorted_people}

    def _sample_and_predict(
        self,
        frame: np.ndarray,
        current_time: float,
        allowed_individual_ids: set[int],
        active_group_states: dict[int, GroupROIState],
    ) -> None:
        ready_states, ready_clips = self._collect_ready_action_clips(
            frame,
            current_time,
            allowed_individual_ids,
            active_group_states,
        )
        worker = self._action_worker
        if not ready_clips or worker is None:
            return
        if current_time + 1e-9 < getattr(self, "_next_action_submit_at", 0.0):
            for state in ready_states:
                state.samples_since_prediction = max(
                    state.samples_since_prediction,
                    PREDICT_EVERY_SAMPLES,
                )
            self._rate_limited_submissions = getattr(self, "_rate_limited_submissions", 0) + 1
            return

        if getattr(self, "_roi_batch_mode", False):
            classifier = self._tcclip_classifier
            if classifier is None:
                return
            submitted_at = time.monotonic()
            if not hasattr(self, "_roi_last_submitted_at"):
                self._roi_last_submitted_at = {}
            group_limit = worker.admission_limit()
            selected = self._select_batch_ready_rois(
                ready_states, ready_clips, submitted_at, limit=group_limit
            )
            selected_ids = {id(state) for state, _ in selected}
            for state in ready_states:
                if id(state) not in selected_ids:
                    state.samples_since_prediction = max(
                        state.samples_since_prediction, PREDICT_EVERY_SAMPLES
                    )
            submitted_any = False
            for state, clip in selected:
                task = _TcClipActionTask(
                    source_frame_id=self._current_frame_id,
                    source_captured_at=getattr(self, "_current_capture_at", submitted_at),
                    submitted_at=submitted_at,
                    generation=self._action_generation,
                    states=(state,),
                    clips=(tuple(clip),),
                )
                state_key = self._state_key(state)
                key = (self._action_queue_key, *state_key)
                with self._action_roi_keys_lock:
                    if not self._action_recognition_enabled:
                        task_id = None
                    else:
                        task_id = worker.submit(key, TcClipRoiBatchJob(classifier, task))
                        if task_id is not None:
                            self._action_roi_keys.add(key)
                if task_id is None:
                    state.samples_since_prediction = max(
                        state.samples_since_prediction, PREDICT_EVERY_SAMPLES
                    )
                    if self._action_recognition_enabled:
                        self._admission_rejections = getattr(self, "_admission_rejections", 0) + 1
                else:
                    submitted_any = True
                    self._roi_last_submitted_at[state_key] = submitted_at
            interval = max(0.0, float(getattr(self, "_inference_interval_sec", 0.0)))
            if submitted_any:
                self._next_action_submit_at = current_time + interval
            elif selected:
                # The camera already has work in flight. Keep sampling frames,
                # but avoid retrying the same admission on every video frame.
                self._next_action_submit_at = current_time + min(0.25, max(0.05, interval / 4))
            return

        selected = self._select_ready_rois(ready_states, ready_clips)
        selected_ids = {id(state) for state, _ in selected}
        for state in ready_states:
            if id(state) not in selected_ids:
                state.samples_since_prediction = max(
                    state.samples_since_prediction,
                    PREDICT_EVERY_SAMPLES,
                )
        if not selected:
            return
        selected_states, selected_clips = zip(*selected, strict=True)
        submitted_at = time.monotonic()
        task = _TcClipActionTask(
            source_frame_id=self._current_frame_id,
            source_captured_at=getattr(self, "_current_capture_at", submitted_at),
            submitted_at=submitted_at,
            generation=self._action_generation,
            states=tuple(selected_states),
            clips=tuple(tuple(clip) for clip in selected_clips),
        )
        task_id = worker.submit(
            self._action_queue_key,
            lambda: self._run_action_task(task),
        )
        if task_id is None:
            for state in selected_states:
                state.samples_since_prediction = max(
                    state.samples_since_prediction,
                    PREDICT_EVERY_SAMPLES,
                )
            return
        self._next_action_submit_at = current_time + max(
            0.0,
            float(getattr(self, "_inference_interval_sec", 0.0)),
        )

    def _select_batch_ready_rois(
        self,
        states: Sequence[BaseROIState],
        clips: Sequence[Sequence[np.ndarray]],
        now: float,
        *,
        limit: int | None = None,
    ) -> list[tuple[BaseROIState, Sequence[np.ndarray]]]:
        """Give each ROI a turn while allowing current alerts to refresh faster."""
        last_submitted = getattr(self, "_roi_last_submitted_at", {})
        priority_names = set(getattr(self, "_alert_classes", set())) | set(
            getattr(self, "_autotrack_classes", set())
        )

        def sort_key(item: tuple[BaseROIState, Sequence[np.ndarray]]) -> tuple[float, int, int]:
            state = item[0]
            last_at = last_submitted.get(self._state_key(state))
            age = float("inf") if last_at is None else max(0.0, now - last_at)
            prediction = state.prediction
            priority = prediction is not None and prediction.action in priority_names
            return (
                -age * (1.5 if priority else 1.0),
                0 if isinstance(state, GroupROIState) else 1,
                state.roi_id,
            )

        candidates = sorted(zip(states, clips, strict=True), key=sort_key)
        max_rois = max(1, int(getattr(self, "_max_rois_per_job", 2)))
        if limit is not None:
            max_rois = min(max_rois, max(1, int(limit)))
        return candidates[:max_rois]

    def _select_ready_rois(
        self,
        states: Sequence[BaseROIState],
        clips: Sequence[Sequence[np.ndarray]],
    ) -> list[tuple[BaseROIState, Sequence[np.ndarray]]]:
        """Choose a bounded rotating ROI subset so crowded frames stay live."""
        candidates = list(zip(states, clips, strict=True))
        if not candidates:
            return []

        def is_priority(item: tuple[BaseROIState, Sequence[np.ndarray]]) -> bool:
            prediction = item[0].prediction
            return prediction is not None and prediction.action in getattr(self, "_autotrack_classes", set())

        def sort_key(item: tuple[BaseROIState, Sequence[np.ndarray]]) -> tuple[int, int]:
            return (0 if isinstance(item[0], GroupROIState) else 1, item[0].roi_id)

        priority = sorted((item for item in candidates if is_priority(item)), key=sort_key)
        regular = sorted((item for item in candidates if not is_priority(item)), key=sort_key)
        selected: list[tuple[BaseROIState, Sequence[np.ndarray]]] = []
        if priority:
            start = getattr(self, "_priority_roi_cursor", 0) % len(priority)
            rotated = priority[start:] + priority[:start]
            max_rois = max(1, int(getattr(self, "_max_rois_per_job", 2)))
            priority_limit = max_rois - (1 if regular and max_rois > 1 else 0)
            chosen = rotated[:priority_limit]
            selected.extend(chosen)
            self._priority_roi_cursor = (start + len(chosen)) % len(priority)
        max_rois = max(1, int(getattr(self, "_max_rois_per_job", 2)))
        remaining = max_rois - len(selected)
        if regular and remaining > 0:
            start = getattr(self, "_roi_cursor", 0) % len(regular)
            rotated = regular[start:] + regular[:start]
            chosen = rotated[:remaining]
            selected.extend(chosen)
            self._roi_cursor = (start + len(chosen)) % len(regular)
        return selected

    def _run_action_task(self, task: _TcClipActionTask) -> _TcClipActionOutput:
        classifier = self._tcclip_classifier
        if classifier is None:
            raise RuntimeError("TC-CLIP classifier is not loaded")
        probabilities, prompts = self._predict_direct_tcclip(task.clips)
        return _TcClipActionOutput(
            task=task,
            probabilities=probabilities,
            prompts=prompts,
        )

    def _consume_action_result(self) -> None:
        worker = self._action_worker
        if worker is None:
            return
        if not getattr(self, "_roi_batch_mode", False):
            result = worker.poll_latest(self._action_queue_key)
            if result is not None:
                self._apply_action_result(result)
            return

        current_keys = {(self._action_queue_key, "individual", track_id) for track_id in self._individual_states} | {
            (self._action_queue_key, "group", roi_id) for roi_id in self._group_states
        }
        with self._action_roi_keys_lock:
            obsolete_keys = self._action_roi_keys - current_keys
            self._action_roi_keys.intersection_update(current_keys)
            active_keys = tuple(self._action_roi_keys)
        for key in obsolete_keys:
            worker.discard(key)
            getattr(self, "_roi_last_submitted_at", {}).pop((key[1], key[2]), None)
        updates = list(self._completed_action_updates)
        for key in active_keys:
            result = worker.poll_latest(key)
            if result is None:
                continue
            self._completed_action_updates = []
            self._apply_action_result(result)
            updates.extend(self._completed_action_updates)
        self._completed_action_updates = updates

    def _apply_action_result(self, result) -> None:
        if result.error is not None:
            self._record_tcclip_error(result.error, context="asynchronous action inference")
            return

        output = result.value
        if output is None:
            return
        if output.task.generation != self._action_generation or output.prompts != self._tcclip_prompts:
            self._stale_action_results += 1
            return
        now = time.monotonic()
        if now - output.task.submitted_at > float(
            getattr(self, "_max_result_age_sec", 2.0)
        ) or now - output.task.source_captured_at > float(getattr(self, "_max_result_age_sec", 2.0)):
            self._stale_action_results += 1
            return
        if len(output.task.states) != len(output.probabilities):
            self._record_tcclip_error(
                RuntimeError("TC-CLIP action result size does not match the submitted ROI count"),
                context="asynchronous action inference",
            )
            return

        valid_states: list[BaseROIState] = []
        valid_probabilities: list[np.ndarray] = []
        for state, probabilities in zip(
            output.task.states,
            output.probabilities,
            strict=True,
        ):
            if self._state_is_current(state):
                valid_states.append(state)
                valid_probabilities.append(probabilities)
        if not valid_states:
            self._stale_action_results += 1
            return

        self._prediction_prompts = output.prompts
        self._apply_action_probabilities(
            valid_states,
            np.stack(valid_probabilities, axis=0),
        )
        action_last_applied = getattr(self, "_action_last_applied", None)
        if action_last_applied is None:
            action_last_applied = {}
            self._action_last_applied = action_last_applied
        for state in valid_states:
            action_last_applied[self._state_key(state)] = now
        self._publish_action_batch(
            valid_states,
            source_frame_id=output.task.source_frame_id,
        )
        self._tcclip_last_error = None

    @staticmethod
    def _state_key(state: BaseROIState) -> tuple[str, int]:
        if isinstance(state, IndividualROIState):
            return "individual", state.track_id
        return "group", state.roi_id

    def _expire_action_labels(self) -> None:
        """Drop action labels too old to drive an alert or the N8 gimbal."""
        now = time.monotonic()
        action_last_applied = getattr(self, "_action_last_applied", {})
        expired = [
            key
            for key, applied_at in action_last_applied.items()
            if now - applied_at > float(getattr(self, "_action_ttl_sec", 2.0))
        ]
        for kind, state_id in expired:
            if kind == "individual":
                state = self._individual_states.get(state_id)
                self._actions.pop(state_id, None)
            else:
                state = self._group_states.get(state_id)
            if state is not None:
                state.prediction = None
                state.probability_history.clear()
            action_last_applied.pop((kind, state_id), None)

    def _state_is_current(self, state: BaseROIState) -> bool:
        if isinstance(state, IndividualROIState):
            return self._individual_states.get(state.track_id) is state
        if isinstance(state, GroupROIState):
            return self._group_states.get(state.roi_id) is state
        return False

    def _discard_action_jobs(self) -> None:
        worker = self._action_worker
        if worker is None:
            return
        if getattr(self, "_roi_batch_mode", False):
            with self._action_roi_keys_lock:
                keys = tuple(self._action_roi_keys)
                self._action_roi_keys.clear()
            for key in keys:
                worker.discard(key)
        else:
            worker.discard(self._action_queue_key)

    def set_feature_flags(
        self,
        *,
        action_recognition_enabled: bool = True,
        object_detection_enabled: bool = True,
    ) -> None:
        was_action_enabled = self._action_recognition_enabled
        super().set_feature_flags(
            action_recognition_enabled=action_recognition_enabled,
            object_detection_enabled=object_detection_enabled,
        )
        if was_action_enabled and not self._action_recognition_enabled:
            self._action_generation += 1
            self._completed_action_updates = []
            self._discard_action_jobs()
            getattr(self, "_roi_last_submitted_at", {}).clear()

    def _predict_action_batch(self, clips: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        if self._tcclip_classifier is None:
            return self._baseline_probabilities(len(clips), self._tcclip_prompts)
        try:
            probabilities, prompts = self._predict_direct_tcclip(clips)
        except Exception as exc:
            self._record_tcclip_error(exc, context="action inference")
            return self._baseline_probabilities(len(clips), self._tcclip_prompts)
        self._tcclip_last_error = None
        self._prediction_prompts = prompts
        if prompts != self._tcclip_prompts:
            self._apply_prompt_metadata(prompts, clear_temporal_state=True)
        return probabilities

    def _baseline_probabilities(self, batch_size: int, prompts: TcClipPrompts) -> np.ndarray:
        probabilities = np.zeros((batch_size, len(prompts.classes)), dtype=np.float32)
        if batch_size > 0:
            baseline_index = prompts.names.index(prompts.baseline_name)
            probabilities[:, baseline_index] = 1.0
        self._prediction_prompts = prompts
        return probabilities

    def _record_tcclip_error(self, exc: Exception, *, context: str) -> None:
        self._tcclip_last_error = str(exc)
        now = time.monotonic()
        if now - self._tcclip_last_error_log_at < TC_CLIP_ERROR_LOG_INTERVAL_SEC:
            return
        self._tcclip_last_error_log_at = now
        logger.error(
            "TC-CLIP %s failed; continuing with YOLO detections and baseline actions: %s",
            context,
            exc,
        )

    @staticmethod
    def _mask_disabled_probabilities(probabilities: np.ndarray, prompts: TcClipPrompts) -> np.ndarray | None:
        """Drop disabled classes and renormalise the rest.

        Renormalising softmax probabilities without some candidates equals
        running the softmax without them, so a disabled class is effectively
        switched off without touching prompts, engines or grid classes. Returns
        None when the disabled classes hold (almost) all the probability: no
        enabled class was seen, and promoting noise would raise false alerts.
        Validation keeps at least two classes enabled, so a single class never
        reaches 100% by default.
        """
        enabled = np.asarray(prompts.enabled_flags, dtype=bool)
        masked = np.where(enabled, probabilities, 0.0).astype(np.float32)
        remaining = float(masked.sum())
        if remaining <= 1e-9:
            return None
        return masked / remaining

    def _choose_action_prediction(self, probabilities: np.ndarray, roi_type: str) -> ActionPrediction:
        del roi_type
        prompts = self._prediction_prompts
        expected_classes = len(prompts.classes)
        if probabilities.shape != (expected_classes,):
            raise ValueError(
                f"TC-CLIP prediction must contain {expected_classes} class probabilities; "
                f"got shape {probabilities.shape}"
            )
        if not all(prompts.enabled_flags):
            masked = self._mask_disabled_probabilities(probabilities, prompts)
            if masked is None:
                # Only disabled classes scored: report no action, never an alert.
                return ActionPrediction(
                    class_name=prompts.baseline_name,
                    display_name=prompts.baseline_name,
                    confidence=0.0,
                    probabilities=np.where(prompts.enabled_flags, probabilities, 0.0).astype(np.float32),
                )
            probabilities = masked
        class_index = int(np.argmax(probabilities))
        class_name = prompts.names[class_index]
        confidence = float(probabilities[class_index])
        display_name = (
            prompts.baseline_name
            if class_name in prompts.alert_names and confidence < self._tcclip_alerts_threshold
            else class_name
        )
        return ActionPrediction(
            class_name=class_name,
            display_name=display_name,
            confidence=confidence,
            probabilities=probabilities.copy(),
        )

    def update_action_prompts(
        self,
        prompt_classes: Sequence[dict[str, object]] | str,
        fight_prompt: str | None = None,
    ) -> None:
        """Update dynamic TC-CLIP candidates without reloading checkpoint weights."""
        prompts = (
            TcClipPrompts(prompt_classes, fight_prompt)
            if isinstance(prompt_classes, str)
            else TcClipPrompts(classes=prompt_classes)
        )
        if self._tcclip_classifier is None:
            raise RuntimeError("TC-CLIP classifier is not loaded")
        self._action_generation += 1
        self._completed_action_updates = []
        self._discard_action_jobs()
        self._tcclip_classifier.update_prompts(prompts)
        self._tcclip_prompts = prompts
        self._select_direct_tcclip_engine()
        self._apply_prompt_metadata(prompts, clear_temporal_state=True)

    def reset_tracking_state(self) -> None:
        """Reject in-flight action work and reset state for a new RTSP session."""
        self._action_generation += 1
        self._completed_action_updates = []
        self._discard_action_jobs()
        getattr(self, "_action_last_applied", {}).clear()
        self._next_action_submit_at = 0.0
        self._roi_cursor = 0
        self._priority_roi_cursor = 0
        getattr(self, "_roi_last_submitted_at", {}).clear()
        super().reset_tracking_state()

    def close(self, *, preserve_yolo_state: bool = False) -> None:
        worker = self._action_worker
        self._discard_action_jobs()
        self._action_worker = None
        if worker is not None and self._owns_action_worker:
            worker.close(timeout=1.0)
        super().close(preserve_yolo_state=preserve_yolo_state)

    def _apply_prompt_metadata(
        self,
        prompts: TcClipPrompts,
        *,
        clear_temporal_state: bool,
    ) -> None:
        """Keep detector state aligned with the shared classifier prompt snapshot."""
        self._tcclip_prompts = prompts
        self._prediction_prompts = prompts
        self._tsm_classes = list(prompts.names)
        self.set_alert_classes_with_levels({
            item.name: item.threat_level for item in prompts.classes if item.is_alert
        })
        if not clear_temporal_state:
            return
        getattr(self, "_action_last_applied", {}).clear()
        for state in self._individual_states.values():
            state.clear_temporal_state()
        for state in self._group_states.values():
            state.clear_temporal_state()
        self._actions.clear()

    def _tcclip_engine_build_status(self) -> dict[str, object]:
        from app.services.detection.tcclip_engine_manager import tcclip_engine_manager

        return tcclip_engine_manager.status(self._tcclip_prompts)

    def get_info(self) -> dict:
        classifier_info = self._tcclip_classifier.get_info() if self._tcclip_classifier else {}
        worker_info = (
            self._action_worker.stats()
            if self._action_worker
            else {
                "closed": True,
                "thread_alive": False,
            }
        )
        return {
            "backend": "yolo_tcclip",
            "mode": "dynamic_roi_zero_shot_multiclass",
            "yolo_model": self._yolo_path,
            "tcclip": classifier_info,
            "prompt_classes": self._tcclip_prompts.to_payload(),
            "alert_classes": sorted(self._alert_classes),
            "alert_threshold": self._tcclip_alerts_threshold,
            "num_segments": self._num_segments,
            "temporal_stride": self._temporal_stride,
            "target_fps": self._target_fps,
            "device": str(self._device),
            "yolo_imgsz": self._yolo_imgsz,
            "yolo_half": self._yolo_half,
            "yolo_inference_engine": (
                getattr(self._yolo_scheduler, "backend", "pytorch")
                if self._yolo_scheduler is not None
                else getattr(self, "_yolo_backend", "pytorch")
            ),
            "tcclip_inference_engine": (
                "tensorrt" if getattr(self, "_tcclip_auto_runner", None) is not None else "pytorch"
            ) if not self._roi_batch_mode else "shared",
            "tcclip_auto_error": getattr(self, "_tcclip_auto_error", None),
            "tcclip_engine_build": self._tcclip_engine_build_status(),
            "loaded": self._yolo_ready() and self._tcclip_classifier is not None,
            "shared_yolo": self._yolo_scheduler is not None,
            "action_degraded": self._tcclip_classifier is None or self._tcclip_last_error is not None,
            "last_error": self._tcclip_last_error,
            "action_worker": worker_info,
            "roi_job_mode": "per_roi_fair_batch" if self._roi_batch_mode else "per_camera",
            "stale_action_results": self._stale_action_results,
            "rate_limited_submissions": self._rate_limited_submissions,
            "admission_rejections": getattr(self, "_admission_rejections", 0),
            "inference_interval_sec": self._inference_interval_sec,
            "max_rois_per_job": self._max_rois_per_job,
            "max_result_age_sec": self._max_result_age_sec,
            "action_ttl_sec": self._action_ttl_sec,
            "active_individual_rois": len(self._individual_states),
            "active_group_rois": len(self._group_states),
            "action_recognition_enabled": self._action_recognition_enabled,
            "object_detection_enabled": self._object_detection_enabled,
        }
