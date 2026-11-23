"""Shared PyTorch/TensorRT aggregation of video-conditioned prompt logits."""

from __future__ import annotations

import torch

from app.services.detection.tcclip_prompts import TcClipPrompts


def aggregate_tcclip_logits(logits: torch.Tensor, prompts: TcClipPrompts) -> torch.Tensor:
    """Mean within each class BEFORE softmax, never across videos or tiles."""
    if logits.ndim != 2 or logits.shape[1] != prompts.prompt_count:
        raise RuntimeError(f"TC-CLIP expected {prompts.prompt_count} prompt logits, got {tuple(logits.shape)}")
    if prompts.prompt_count == len(prompts.classes):
        return logits
    return torch.stack([logits[:, start:end].mean(dim=1) for start, end in prompts.prompt_ranges], dim=1)
