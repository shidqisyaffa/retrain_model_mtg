"""Validate TC-CLIP's effective BPE budget without importing its model runtime."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.detection.tcclip_prompts import prompt_variant_rows

TC_CLIP_CONTEXT_LENGTH = 30
TC_CLIP_PROMPT_PREFIX = "a photo of a"


@lru_cache(maxsize=4)
def _load_prompt_tokenizer(runtime_root: str) -> Any:
    """Load only SimpleTokenizer; never import clip/__init__.py or torch."""
    clip_dir = Path(runtime_root) / "clip"
    source = clip_dir / "simple_tokenizer.py"
    vocabulary = clip_dir / "bpe_simple_vocab_16e6.txt.gz"
    if not source.is_file() or not vocabulary.is_file():
        raise ValueError(
            f"TC-CLIP prompt tokenizer is incomplete at {clip_dir}; "
            "verify TC_CLIP_RUNTIME_ROOT and the vendored runtime."
        )
    spec = importlib.util.spec_from_file_location("_tcclip_prompt_tokenizer", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load TC-CLIP prompt tokenizer from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleTokenizer(str(vocabulary))


def validate_tcclip_prompt_tokens(
    prompt_classes: Sequence[Mapping[str, object]],
    *,
    runtime_root: str | Path,
    context_length: int = TC_CLIP_CONTEXT_LENGTH,
    prompt_prefix: str = TC_CLIP_PROMPT_PREFIX,
) -> dict[str, int]:
    """Reject overflow before Apply; count the template, period, SOS and EOT.

    This mirrors VPPromptLearner's English class text construction. It does
    not truncate input, translate text, load weights, or change active prompts.
    """
    if context_length < 2:
        raise ValueError("TC-CLIP text context length must be at least 2")
    if not prompt_classes:
        return {}
    tokenizer = _load_prompt_tokenizer(str(Path(runtime_root).expanduser().resolve()))
    counts: dict[str, int] = {}
    for item in prompt_classes:
        if not isinstance(item, Mapping):
            raise ValueError("Each TC-CLIP class must be an object")
        name = str(item.get("name") or item.get("class_name") or "").strip().casefold()
        variants = prompt_variant_rows(item)
        for index, variant in enumerate(variants):
            label = f"class '{name}'" + (f" variant {index + 1}" if len(variants) > 1 else "")
            prompt = str(variant.get("translated_prompt") or variant.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"TC-CLIP prompt for {label} must not be empty")
            rendered = prompt_prefix + " " + prompt.replace("_", " ") + "."
            count = len(tokenizer.encode(rendered)) + 2
            if count > context_length:
                guidance = (
                    "Shorten the Indonesian source prompt and apply again"
                    if variant.get("source_language") == "id" else "Shorten the English prompt"
                )
                raise ValueError(
                    f"TC-CLIP prompt for {label} uses {count} tokens including "
                    f"the template and SOS/EOT; maximum is {context_length}. "
                    f"{guidance}; text is not automatically truncated."
                )
            counts[f"{name}[{index + 1}]" if len(variants) > 1 else name] = count
    return counts
