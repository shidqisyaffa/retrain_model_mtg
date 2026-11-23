"""Lightweight validated TC-CLIP prompt contracts (no model/runtime imports)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TC_CLIP_MAX_CLASSES = 32
TC_CLIP_MAX_PROMPTS = 32


def prompt_variant_rows(item: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Normalize nested variants without changing the legacy one-string contract."""
    if "prompts" not in item:
        return (item,)
    variants = item["prompts"]
    name = str(item.get("name") or item.get("class_name") or "")
    if not isinstance(variants, (list, tuple)) or not variants:
        raise ValueError(f"TC-CLIP prompts for class '{name}' must be a non-empty array")
    if len(variants) > TC_CLIP_MAX_PROMPTS:
        raise ValueError(f"TC-CLIP supports at most {TC_CLIP_MAX_PROMPTS} total prompt variants")
    rows = []
    for variant in variants:
        if isinstance(variant, str):
            variant = {"prompt": variant}
        if not isinstance(variant, Mapping):
            raise ValueError(f"Each TC-CLIP prompt variant for class '{name}' must be an object or string")
        rows.append({"source_language": item.get("source_language") or "en", **variant})
    return tuple(rows)


@dataclass(frozen=True)
class TcClipPromptVariant:
    """One effective English text plus its independent translation provenance."""

    prompt: str
    source_language: str = "en"
    source_prompt: str = ""
    translation_edited: bool = False

    def __post_init__(self) -> None:
        prompt = self.prompt.strip()
        source = self.source_prompt.strip() or prompt
        language = self.source_language.strip().lower() or "en"
        if not prompt:
            raise ValueError("TC-CLIP prompt must not be empty")
        if len(prompt) > 1000 or len(source) > 1000:
            raise ValueError("TC-CLIP prompt must not exceed 1000 characters")
        if language not in {"en", "id"}:
            raise ValueError("TC-CLIP source_language must be 'id' or 'en'")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "source_prompt", source)
        object.__setattr__(self, "source_language", language)
        object.__setattr__(self, "translation_edited", bool(self.translation_edited))

    def to_payload(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "source_language": self.source_language,
            "source_prompt": self.source_prompt,
            "translated_prompt": self.prompt,
            "translation_edited": self.translation_edited,
        }


@dataclass(frozen=True)
class TcClipPromptClass:
    """One action with one or more texts and class-level alert behavior."""

    name: str
    prompt: str
    is_alert: bool = False
    source_language: str = "en"
    source_prompt: str = ""
    translation_edited: bool = False
    auto_track: bool = False
    threat_level: str = "high"
    prompts: tuple[TcClipPromptVariant, ...] = ()
    # A disabled class is ignored by the person/union action detector when it
    # picks the winning class. Its text stays a model candidate, so engines and
    # the shared classifier are unaffected.
    enabled: bool = True

    def __post_init__(self) -> None:
        name = self.name.strip().casefold()
        try:
            variants = tuple(self.prompts) or (
                TcClipPromptVariant(self.prompt, self.source_language, self.source_prompt, self.translation_edited),
            )
        except ValueError as exc:
            raise ValueError(f"{exc} for class '{name}'") from exc
        if not all(isinstance(item, TcClipPromptVariant) for item in variants):
            raise ValueError(f"Invalid TC-CLIP prompt variants for class '{name}'")
        first = variants[0]
        threat_level = str(self.threat_level).strip().lower()
        if threat_level not in {"medium", "high"}:
            raise ValueError("TC-CLIP threat_level must be 'medium' or 'high'")
        if not name:
            raise ValueError("TC-CLIP class name must not be empty")
        if len(name) > 64:
            raise ValueError("TC-CLIP class name must not exceed 64 characters")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "prompt", first.prompt)
        object.__setattr__(self, "is_alert", bool(self.is_alert))
        object.__setattr__(self, "source_language", first.source_language)
        object.__setattr__(self, "source_prompt", first.source_prompt)
        object.__setattr__(self, "translation_edited", first.translation_edited)
        object.__setattr__(self, "prompts", variants)
        object.__setattr__(self, "auto_track", bool(self.auto_track))
        object.__setattr__(self, "threat_level", threat_level)
        object.__setattr__(self, "enabled", bool(self.enabled))


@dataclass(frozen=True, init=False)
class TcClipPrompts:
    """Validated dynamic class list used by the TC-CLIP backend."""

    classes: tuple[TcClipPromptClass, ...]

    def __init__(
        self,
        normal: str | None = None,
        fight: str | None = None,
        *,
        classes: Sequence[TcClipPromptClass | Mapping[str, Any]] | None = None,
    ) -> None:
        if classes is None:
            candidates: Sequence[TcClipPromptClass | Mapping[str, Any]] = (
                TcClipPromptClass("normal", normal or "", False),
                TcClipPromptClass("fight", fight or "", True),
            )
        else:
            candidates = classes

        parsed: list[TcClipPromptClass] = []
        for item in candidates:
            if isinstance(item, TcClipPromptClass):
                parsed.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError("Each TC-CLIP class must be an object")
            variants = []
            for index, row in enumerate(prompt_variant_rows(item)):
                try:
                    variants.append(TcClipPromptVariant(
                        prompt=str(row.get("translated_prompt") or row.get("prompt") or ""),
                        source_language=str(row.get("source_language") or "en"),
                        source_prompt=str(row.get("source_prompt") or row.get("prompt") or ""),
                        translation_edited=bool(row.get("translation_edited", False)),
                    ))
                except ValueError as exc:
                    raise ValueError(f"{exc} for class '{item.get('name', '')}', variant {index + 1}") from exc
            parsed.append(
                TcClipPromptClass(
                    name=str(item.get("name") or item.get("class_name") or ""),
                    prompt=str(item.get("translated_prompt") or item.get("prompt") or ""),
                    is_alert=bool(item.get("is_alert", False)),
                    source_language=str(item.get("source_language") or "en"),
                    source_prompt=str(item.get("source_prompt") or item.get("prompt") or ""),
                    translation_edited=bool(item.get("translation_edited", False)),
                    auto_track=bool(item.get("auto_track", False)),
                    threat_level=str(item.get("threat_level") or "high"),
                    prompts=tuple(variants),
                    enabled=item.get("enabled") is not False,
                )
            )

        if len(parsed) < 2:
            raise ValueError("TC-CLIP requires at least two prompt classes")
        if len(parsed) > TC_CLIP_MAX_CLASSES:
            raise ValueError(f"TC-CLIP supports at most {TC_CLIP_MAX_CLASSES} prompt classes")
        names = [item.name.casefold() for item in parsed]
        if len(set(names)) != len(names):
            raise ValueError("TC-CLIP class names must be unique")
        prompt_values = [variant.prompt.casefold() for item in parsed for variant in item.prompts]
        if len(prompt_values) > TC_CLIP_MAX_PROMPTS:
            raise ValueError(f"TC-CLIP supports at most {TC_CLIP_MAX_PROMPTS} total prompt variants across all classes")
        seen_prompts: dict[str, tuple[str, int]] = {}
        for item in parsed:
            for index, variant in enumerate(item.prompts, 1):
                key = variant.prompt.casefold()
                if key in seen_prompts:
                    previous_name, previous_index = seen_prompts[key]
                    raise ValueError(
                        f"TC-CLIP prompt texts must be different: class '{item.name}' variant {index} "
                        f"duplicates class '{previous_name}' variant {previous_index}"
                    )
                seen_prompts[key] = (item.name, index)
        baseline = next((item for item in parsed if item.name == "normal"), None)
        if baseline is None:
            raise ValueError("TC-CLIP requires a baseline class named 'normal'")
        if baseline.is_alert:
            raise ValueError("TC-CLIP baseline class 'normal' must not be an alert")
        if not any(item.is_alert for item in parsed):
            raise ValueError("TC-CLIP requires at least one alert class")
        enabled = [item for item in parsed if item.enabled]
        if len(enabled) < 2:
            raise ValueError("TC-CLIP requires at least two enabled classes")
        if not any(item.is_alert for item in enabled):
            raise ValueError("TC-CLIP requires at least one enabled alert class")
        object.__setattr__(self, "classes", tuple(parsed))

    @property
    def values(self) -> tuple[str, ...]:
        """Flatten model text candidates; public output names remain class-based."""
        return tuple(variant.prompt for item in self.classes for variant in item.prompts)

    @property
    def prompt_count(self) -> int:
        return sum(len(item.prompts) for item in self.classes)

    @property
    def prompt_ranges(self) -> tuple[tuple[int, int], ...]:
        ranges = []
        start = 0
        for item in self.classes:
            end = start + len(item.prompts)
            ranges.append((start, end))
            start = end
        return tuple(ranges)

    @property
    def candidate_to_class(self) -> tuple[int, ...]:
        return tuple(index for index, item in enumerate(self.classes) for _ in item.prompts)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)

    @property
    def alert_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.classes if item.is_alert)

    @property
    def baseline_name(self) -> str:
        return "normal"

    @property
    def enabled_flags(self) -> tuple[bool, ...]:
        """One flag per class, in class order; False means action choice ignores it."""
        return tuple(item.enabled for item in self.classes)

    @property
    def normal(self) -> str:
        item = next((item for item in self.classes if item.name == "normal"), self.classes[0])
        return item.prompt

    @property
    def fight(self) -> str:
        item = next(
            (item for item in self.classes if item.name == "fight"),
            next((item for item in self.classes if item.is_alert), self.classes[1]),
        )
        return item.prompt

    def to_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "prompt": item.prompt,
                "is_alert": item.is_alert,
                "source_language": item.source_language,
                "source_prompt": item.source_prompt,
                "translated_prompt": item.prompt,
                "translation_edited": item.translation_edited,
                "auto_track": item.auto_track,
                "threat_level": item.threat_level,
                **({"enabled": False} if not item.enabled else {}),
                **({"prompts": [variant.to_payload() for variant in item.prompts]} if len(item.prompts) > 1 else {}),
            }
            for item in self.classes
        ]
