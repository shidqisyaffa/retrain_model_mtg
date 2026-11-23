"""Stable TC-CLIP token merge for comparing PyTorch with TensorRT.

Adapted from the upstream ToMe merge implementation bundled with TC-CLIP.
The upstream implementation is Copyright (c) Meta Platforms, Inc. and
affiliates, and remains subject to its bundled LICENSE.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable

import torch
from torch.nn import functional as F

MERGE_VERSION = "tome-one-hot-fp32-v1"


def stable_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> tuple[Callable, Callable]:
    protected = int(class_token) + int(distill_token)
    r = min(r, (metric.shape[1] - protected) // 2)
    if r <= 0:
        return lambda x, mode=None: x, lambda x: x

    with torch.no_grad():
        normalized = metric / metric.norm(dim=-1, keepdim=True)
        a, b = normalized[..., ::2, :], normalized[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        if mode == "sum":
            # FP32 accumulation with explicit destinations avoids the
            # non-deterministic GPU atomic additions in scatter_reduce.
            assignment = F.one_hot(dst_idx[..., 0], num_classes=dst.size(-2))
            updates = torch.bmm(assignment.transpose(1, 2).float(), src.float())
            dst = (dst.float() + updates).to(dst.dtype)
        else:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


def install_stable_merge() -> None:
    """Replace the alias imported by TC-CLIP before creating its model."""
    blocks = importlib.import_module("clip.transformer_blocks_tc")
    blocks.bipartite_soft_matching = stable_bipartite_soft_matching
