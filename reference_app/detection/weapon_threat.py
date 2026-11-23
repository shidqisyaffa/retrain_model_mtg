"""Threat level for a person holding a knife/weapon."""

from __future__ import annotations

from app.core.config.config import settings

_RANK = {"low": 1, "medium": 2, "high": 3}


def weapon_threat_level(action_level: str | None = None) -> str:
    """Level for a knife/weapon detection.

    Comes from ``detection.weapon_threat_level`` (default "medium": small objects are
    easy to misdetect, so the operator reviews before guards are sent). Never lower
    than the person's own action class, e.g. fight (HIGH) + knife stays HIGH.
    """
    base = settings.DETECTION_WEAPON_THREAT_LEVEL
    other = str(action_level or "").strip().lower()
    return other if _RANK.get(other, 0) > _RANK[base] else base
