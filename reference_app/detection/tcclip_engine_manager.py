"""Build one prompt-dynamic TensorRT engine in one background process."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from app.core.config.config import settings
from app.services.detection.tcclip_prompts import TcClipPrompts

logger = logging.getLogger(__name__)
# Uvicorn's default config does not attach an INFO handler to application/root
# loggers. Use its console hierarchy for operator-visible engine selection.
engine_logger = logging.getLogger("uvicorn.error.tcclip")


def log_tcclip_engine_policy() -> None:
    engine_logger.info(
        "[TC-CLIP] policy=auto: prefer compatible TensorRT on NVIDIA CUDA; "
        "otherwise use PyTorch. Saved manual engine preferences are not restored; "
        "effective engines are logged when selected by each runtime."
    )


class TcClipEngineSelectionLog:
    """Report effective engine changes once per runtime, never once per frame."""

    def __init__(self, scope: str) -> None:
        self._scope = scope
        self._state: tuple[str, str, str] | None = None
        self._log_lock = threading.Lock()

    def report(self, engine: str, *, device: object = None, reason: str) -> None:
        state = (engine, str(device) if device is not None else "", reason)
        with self._log_lock:
            if state == self._state:
                return
            self._state = state
            engine_logger.info(
                "[TC-CLIP][%s] engine=%s%s reason=%s",
                self._scope,
                "TensorRT" if engine == "tensorrt" else "PyTorch",
                f" device={device}" if device is not None else "",
                reason,
            )


class TcClipEngineManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._desired: tuple[str, TcClipPrompts] | None = None
        self._state = "idle"
        self._error: str | None = None
        self._generation = 0

    @staticmethod
    def candidate_paths() -> tuple[Path, ...]:
        base = settings.CCTV_TC_CLIP_TENSORRT_ENGINE_PATH
        dynamic = base.with_name(f"{base.stem}_dynamic{base.suffix}")
        return (dynamic,)

    def status(self, prompts: TcClipPrompts) -> dict[str, object]:
        signature = "prompt-dynamic-v2"
        with self._lock:
            active = self._desired is not None and self._desired[0] == signature
            return {
                "state": self._state if active else "idle",
                "error": self._error if active else None,
                "generation": self._generation,
                "engine_path": str(self.candidate_paths()[0]),
            }

    def request(self, prompts: TcClipPrompts, *, force: bool = False) -> None:
        signature = "prompt-dynamic-v2"
        with self._lock:
            if (
                not force
                and self._desired is not None
                and self._desired[0] == signature
                and self._state in {"queued", "building", "ready", "failed"}
            ):
                return
            self._desired = (signature, prompts)
            self._state = "queued"
            self._error = None
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="tcclip-engine-build", daemon=True)
                self._worker.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                assert self._desired is not None
                signature, prompts = self._desired
                self._state = "building"
            try:
                self._build(prompts)
                error = None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("TC-CLIP TensorRT background build failed: %s", error)
            with self._lock:
                self._generation += 1
                if self._desired is not None and self._desired[0] == signature:
                    self._state = "failed" if error else "ready"
                    self._error = error
                    self._worker = None
                    return
                # The operator changed prompts during the build. Build only the
                # latest requested set next; intermediate updates are skipped.

    @staticmethod
    def _build(prompts: TcClipPrompts) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        output = TcClipEngineManager.candidate_paths()[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tcclip-prompts-", dir=output.parent) as directory:
            prompts_file = Path(directory) / "prompts.json"
            prompts_file.write_text(json.dumps(prompts.to_payload(), ensure_ascii=False), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "app.tools.build_tcclip_tensorrt",
                "--batch-size",
                str(settings.CCTV_TC_CLIP_BATCH_SIZE),
                "--output",
                str(output),
                "--prompts-file",
                str(prompts_file),
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=1800,
                creationflags=creationflags,
                check=False,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip().splitlines()[-5:]
                raise RuntimeError(f"engine builder exited {completed.returncode}: {' | '.join(details)}")
            logger.info("TC-CLIP dynamic TensorRT engine ready for 2-32 prompts: %s", output)


tcclip_engine_manager = TcClipEngineManager()
