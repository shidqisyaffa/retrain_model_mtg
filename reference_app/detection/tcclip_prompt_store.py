"""Global active TC-CLIP prompt state: validated runtime + durable SQLite state."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from app.core.config.config import settings
from app.services.detection.tcclip_prompts import TcClipPrompts
from app.services.fleet_store import FleetStore, fleet_store

logger = logging.getLogger(__name__)


class TcClipPromptState:
    """Serialize prompt updates across CCTV/drone channels in one process."""

    def __init__(self, store: FleetStore) -> None:
        self._store = store
        self._lock = threading.RLock()

    @staticmethod
    def _update_settings(prompts: TcClipPrompts) -> None:
        settings.DETECTION_TC_CLIP_PROMPT_CLASSES = prompts.to_payload()
        settings.DETECTION_TC_CLIP_NORMAL_PROMPT = prompts.normal
        settings.DETECTION_TC_CLIP_FIGHT_PROMPT = prompts.fight

    def restore(self) -> bool:
        """Load saved prompts before service creation; keep config defaults on failure."""
        with self._lock:
            try:
                shared = self._store.get_ai_shared_config()
                payload = shared.get("tcclip_prompt_classes") if shared else None
                if not payload:
                    payload = self._store.get_tcclip_prompts()
                if payload is None:
                    return False
                prompts = TcClipPrompts(classes=payload)
            except Exception:
                logger.exception("Unable to restore TC-CLIP prompts; using configured defaults")
                return False
            self._update_settings(prompts)
            logger.info("Restored %d active TC-CLIP prompt classes from database", len(prompts.classes))
            return True

    def adopt(self, prompts: TcClipPrompts, update_runtime: Callable[[TcClipPrompts], None]) -> None:
        """Apply a committed revision from another process without writing it again."""
        with self._lock:
            previous = TcClipPrompts(classes=settings.DETECTION_TC_CLIP_PROMPT_CLASSES)
            try:
                update_runtime(prompts)
            except Exception:
                try:
                    update_runtime(previous)
                except Exception:
                    logger.exception("Failed to roll back shared TC-CLIP runtime prompts")
                raise
            self._update_settings(prompts)

    def apply(self, prompts: TcClipPrompts, update_runtime: Callable[[TcClipPrompts], None]) -> None:
        """Acknowledge only after runtime update and durable commit both succeed."""
        with self._lock:
            previous = TcClipPrompts(classes=settings.DETECTION_TC_CLIP_PROMPT_CLASSES)
            try:
                update_runtime(prompts)
                self._store.save_tcclip_prompts(prompts.to_payload())
            except Exception:
                try:
                    update_runtime(previous)
                except Exception:
                    logger.exception("Failed to roll back TC-CLIP runtime prompts")
                raise
            self._update_settings(prompts)


active_tcclip_prompt_state = TcClipPromptState(fleet_store)
