"""Fair, fresh ROI batching for the shared CCTV TC-CLIP model."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence

from app.services.detection.action_queue import ActionWorkerResult


class KeyedLatestBatchActionWorker[KeyT, GroupT, TaskT, ResultT]:
    """Keep one pending clip per ROI and take turns across camera groups."""

    def __init__(
        self,
        handler: Callable[[Sequence[TaskT]], Sequence[ResultT]],
        *,
        group_of: Callable[[KeyT], GroupT],
        max_batch_size: int,
        batch_window_ms: float,
        max_pending_age_sec: float,
        name: str,
        max_rois_per_group: int | None = None,
        batch_target_latency_sec: float | None = None,
    ) -> None:
        if max_batch_size < 1 or batch_window_ms < 0 or max_pending_age_sec <= 0:
            raise ValueError("Invalid TC-CLIP batch size, window, or pending age")
        if max_rois_per_group is not None and max_rois_per_group < 1:
            raise ValueError("TC-CLIP max_rois_per_group must be positive")
        if batch_target_latency_sec is not None and batch_target_latency_sec <= 0:
            raise ValueError("TC-CLIP batch target latency must be positive")
        self._batch_target_latency_sec = batch_target_latency_sec
        self._target_batch_size = max_batch_size
        self._fast_full_batches = 0
        self._handler = handler
        self._group_of = group_of
        self._max_batch_size = max_batch_size
        self._batch_window_sec = batch_window_ms / 1000.0
        self._max_pending_age_sec = max_pending_age_sec
        self._max_rois_per_group = max_rois_per_group
        self._condition = threading.Condition()
        self._pending: dict[KeyT, tuple[int, float, TaskT]] = {}
        self._group_keys: dict[GroupT, deque[KeyT]] = {}
        self._group_order: deque[GroupT] = deque()
        self._results: dict[KeyT, ActionWorkerResult[ResultT]] = {}
        self._active_ids: dict[KeyT, int] = {}
        self._cancelled_ids: set[int] = set()
        self._closed = False
        self._busy = False
        self._next_task_id = 1
        self._submitted = 0
        self._completed = 0
        self._batches = 0
        self._dropped_pending = 0
        self._expired_pending = 0
        self._dropped_results = 0
        self._failures = 0
        self._last_batch_size = 0
        self._last_queue_wait_ms = 0.0
        self._last_batch_duration_ms = 0.0
        self._estimated_clips_per_sec = 0.0
        self._group_seen_at: dict[GroupT, float] = {}
        self._group_completed: dict[GroupT, int] = {}
        self._group_rejected: dict[GroupT, int] = {}
        self._group_last_completed_at: dict[GroupT, float] = {}
        self._rejected_group_limit = 0
        self._rejected_queue_full = 0
        self._evicted_for_fairness = 0
        self._max_observed_batch_size = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, key: KeyT, task: TaskT) -> int | None:
        with self._condition:
            if self._closed:
                return None
            now = time.monotonic()
            group = self._group_of(key)
            self._group_seen_at[group] = now
            # Replacements remain allowed even when the queue is full: each
            # ROI keeps only its newest clip. At most one GPU batch waits.
            self._expire_pending(now)
            if key not in self._pending and self._max_rois_per_group is not None:
                group_jobs = sum(self._group_of(item) == group for item in self._pending)
                group_jobs += sum(self._group_of(item) == group for item in self._active_ids)
                if group_jobs >= self._group_limit(now):
                    self._rejected_group_limit += 1
                    self._group_rejected[group] = self._group_rejected.get(group, 0) + 1
                    return None
            if key not in self._pending and not self._make_pending_room(group):
                self._rejected_queue_full += 1
                self._group_rejected[group] = self._group_rejected.get(group, 0) + 1
                return None
            task_id = self._next_task_id
            self._next_task_id += 1
            self._submitted += 1
            if key in self._pending:
                self._dropped_pending += 1
            else:
                keys = self._group_keys.get(group)
                if keys is None:
                    keys = deque()
                    self._group_keys[group] = keys
                    self._group_order.append(group)
                keys.append(key)
            self._pending[key] = (task_id, now, task)
            self._condition.notify()
            return task_id

    def _group_limit(self, now: float) -> int:
        assert self._max_rois_per_group is not None
        active_window = max(2.0, self._max_pending_age_sec * 2.0)
        active_groups = sum(now - seen <= active_window for seen in self._group_seen_at.values())
        # A small batch's measured throughput cannot estimate a full batch's
        # capacity. Reserve one active + one pending batch across cameras so
        # raising max_batch_size actually lets the GPU see more candidates.
        capacity = self._target_batch_size * 2
        per_group = (capacity + max(1, active_groups) - 1) // max(1, active_groups)
        return max(1, min(self._max_rois_per_group, per_group))

    def _expire_pending(self, now: float) -> None:
        for key, (_, submitted_at, _) in list(self._pending.items()):
            if now - submitted_at <= self._max_pending_age_sec:
                continue
            self._pending.pop(key)
            self._expired_pending += 1
            group = self._group_of(key)
            keys = self._group_keys[group]
            keys.remove(key)
            if not keys:
                self._group_keys.pop(group)
                self._group_order.remove(group)

    def _make_pending_room(self, group: GroupT) -> bool:
        if len(self._pending) < self._target_batch_size:
            return True
        largest_group = max(self._group_keys, key=lambda item: len(self._group_keys[item]))
        incoming_count = len(self._group_keys.get(group, ()))
        keys = self._group_keys[largest_group]
        if largest_group == group or len(keys) <= max(1, incoming_count):
            return False
        # A crowded camera cannot occupy every pending slot. Preserve each
        # camera's oldest turn while replacing its last queued candidate.
        evicted_key = keys.pop()
        self._pending.pop(evicted_key)
        self._dropped_pending += 1
        self._evicted_for_fairness += 1
        return True

    def _adapt_batch_size(self, size: int, duration: float) -> None:
        latency = self._batch_target_latency_sec
        if latency is None or size < max(1, (self._target_batch_size * 3 + 3) // 4):
            return
        if duration > latency:
            self._fast_full_batches = 0
            self._target_batch_size = max(1, min(self._target_batch_size - 1, int(size * latency / duration)))
            # Resize the waiting batch immediately, retaining one turn per
            # camera wherever possible. Do not carry the old larger backlog.
            while len(self._pending) > self._target_batch_size:
                group = max(self._group_keys, key=lambda item: len(self._group_keys[item]))
                keys = self._group_keys[group]
                self._pending.pop(keys.pop())
                self._dropped_pending += 1
                if not keys:
                    self._group_keys.pop(group)
                    self._group_order.remove(group)
        elif duration < latency * 0.6:
            self._fast_full_batches += 1
            if self._fast_full_batches >= 3:
                self._target_batch_size = min(
                    self._max_batch_size, self._target_batch_size + max(1, self._target_batch_size // 4)
                )
                self._fast_full_batches = 0
        else:
            self._fast_full_batches = 0

    def admission_limit(self) -> int:
        """Maximum ROI candidates one camera should offer this cycle."""
        with self._condition:
            if self._max_rois_per_group is None:
                return self._max_batch_size
            return self._group_limit(time.monotonic())

    def poll_latest(self, key: KeyT) -> ActionWorkerResult[ResultT] | None:
        with self._condition:
            return self._results.pop(key, None)

    def discard(self, key: KeyT) -> None:
        with self._condition:
            active_id = self._active_ids.get(key)
            if active_id is not None:
                self._cancelled_ids.add(active_id)
            if self._pending.pop(key, None) is not None:
                self._dropped_pending += 1
                group = self._group_of(key)
                keys = self._group_keys.get(group)
                if keys is not None:
                    keys.remove(key)
                    if not keys:
                        self._group_keys.pop(group, None)
                        self._group_order.remove(group)
            if self._results.pop(key, None) is not None:
                self._dropped_results += 1

    def stats(self) -> dict[str, object]:
        with self._condition:
            now = time.monotonic()
            active_window = max(2.0, self._max_pending_age_sec * 2.0)
            active_groups = [group for group, seen_at in self._group_seen_at.items() if now - seen_at <= active_window]
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "batches": self._batches,
                "dropped_pending": self._dropped_pending,
                "expired_pending": self._expired_pending,
                "dropped_results": self._dropped_results,
                "failures": self._failures,
                "pending": len(self._pending),
                "completed_pending_delivery": len(self._results),
                "last_batch_size": self._last_batch_size,
                "max_observed_batch_size": self._max_observed_batch_size,
                "last_batch_fill_ratio": round(self._last_batch_size / self._max_batch_size, 3),
                "max_pending": self._target_batch_size,
                "target_batch_size": self._target_batch_size,
                "batch_target_latency_sec": self._batch_target_latency_sec,
                "rejected_queue_full": self._rejected_queue_full,
                "evicted_for_fairness": self._evicted_for_fairness,
                "last_queue_wait_ms": round(self._last_queue_wait_ms, 2),
                "last_batch_duration_ms": round(self._last_batch_duration_ms, 2),
                "estimated_clips_per_sec": round(self._estimated_clips_per_sec, 2),
                "active_groups": len(active_groups),
                "group_admission_limit": (self._group_limit(now) if self._max_rois_per_group is not None else None),
                "rejected_group_limit": self._rejected_group_limit,
                "groups": {
                    str(group): {
                        "pending": sum(self._group_of(key) == group for key in self._pending),
                        "active": sum(self._group_of(key) == group for key in self._active_ids),
                        "completed": self._group_completed.get(group, 0),
                        "rejected": self._group_rejected.get(group, 0),
                        "last_completed_age_ms": (
                            round((now - self._group_last_completed_at[group]) * 1000, 2)
                            if group in self._group_last_completed_at
                            else None
                        ),
                    }
                    for group in active_groups
                },
                "max_batch_size": self._max_batch_size,
                "busy": self._busy,
                "closed": self._closed,
                "thread_alive": self._thread.is_alive(),
            }

    def close(self, *, timeout: float = 1.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending.clear()
            self._group_keys.clear()
            self._group_order.clear()
            self._results.clear()
            self._active_ids.clear()
            self._cancelled_ids.clear()
            self._group_seen_at.clear()
            self._group_completed.clear()
            self._group_rejected.clear()
            self._group_last_completed_at.clear()
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))

    def _take_one(self) -> tuple[KeyT, int, float, TaskT] | None:
        while self._group_order:
            group = self._group_order.popleft()
            keys = self._group_keys.get(group)
            if not keys:
                self._group_keys.pop(group, None)
                continue
            key = keys.popleft()
            if keys:
                self._group_order.append(group)
            else:
                self._group_keys.pop(group, None)
            pending = self._pending.pop(key, None)
            if pending is not None:
                task_id, submitted_at, task = pending
                return key, task_id, submitted_at, task
        return None

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                deadline = time.monotonic() + self._batch_window_sec
                while len(self._pending) < self._target_batch_size and not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # Notifications wake collection, but do not end the window.
                    self._condition.wait(remaining)
                if self._closed:
                    return
                selected = []
                while self._pending and len(selected) < self._target_batch_size:
                    item = self._take_one()
                    if item is None:
                        break
                    if time.monotonic() - item[2] > self._max_pending_age_sec:
                        self._expired_pending += 1
                        continue
                    selected.append(item)
                if not selected:
                    continue
                for key, task_id, _, _ in selected:
                    self._active_ids[key] = task_id
                self._busy = True
                self._last_batch_size = len(selected)
                self._max_observed_batch_size = max(self._max_observed_batch_size, len(selected))
                self._last_queue_wait_ms = (time.monotonic() - min(item[2] for item in selected)) * 1000

            started_at = time.monotonic()
            error: Exception | None = None
            values: Sequence[ResultT] = ()
            try:
                values = self._handler([item[3] for item in selected])
                if len(values) != len(selected):
                    raise RuntimeError("TC-CLIP batch result count does not match ROI count")
            except Exception as exc:
                error = exc
            completed_at = time.monotonic()

            with self._condition:
                self._busy = False
                self._batches += 1
                self._completed += len(selected)
                duration = max(completed_at - started_at, 1e-6)
                self._last_batch_duration_ms = duration * 1000
                if error is None:
                    self._adapt_batch_size(len(selected), duration)
                    observed_rate = len(selected) / duration
                    self._estimated_clips_per_sec = (
                        observed_rate
                        if self._estimated_clips_per_sec == 0.0
                        else 0.75 * self._estimated_clips_per_sec + 0.25 * observed_rate
                    )
                if error is not None:
                    self._failures += len(selected)
                for index, (key, task_id, submitted_at, _) in enumerate(selected):
                    group = self._group_of(key)
                    self._group_completed[group] = self._group_completed.get(group, 0) + 1
                    self._group_last_completed_at[group] = completed_at
                    self._active_ids.pop(key, None)
                    cancelled = task_id in self._cancelled_ids
                    self._cancelled_ids.discard(task_id)
                    if self._closed or cancelled:
                        self._dropped_results += 1
                        continue
                    if key in self._results:
                        self._dropped_results += 1
                    self._results[key] = ActionWorkerResult(
                        task_id=task_id,
                        value=None if error else values[index],
                        error=error,
                        submitted_at=submitted_at,
                        started_at=started_at,
                        completed_at=completed_at,
                    )
                if self._closed:
                    return
