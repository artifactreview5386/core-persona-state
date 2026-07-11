"""Uncertainty features for paper-faithful CORE."""

from __future__ import annotations

import math

import torch


def _masked_normalize(p: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-12) -> torch.Tensor:
    if mask is not None:
        p = p.masked_fill(~mask.bool(), 0.0)
    denom = p.sum(dim=-1, keepdim=True).clamp_min(eps)
    return p / denom


def entropy_normalized(p: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-12) -> torch.Tensor:
    """Return H(p) / log(|V_k|) for batch/slot distributions."""
    p = _masked_normalize(p, mask, eps)
    if mask is None:
        cardinality = torch.full_like(p[..., 0], p.size(-1), dtype=p.dtype)
    else:
        cardinality = mask.bool().sum(dim=-1).to(dtype=p.dtype)
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)
    denom = cardinality.clamp_min(2.0).log()
    out = entropy / denom
    return torch.where(cardinality > 1, out, torch.zeros_like(out))


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Standard Jensen-Shannon divergence, normalized by log(2)."""
    p = _masked_normalize(p, mask, eps)
    q = _masked_normalize(q, mask, eps)
    m = 0.5 * (p + q)
    kl_pm = _kl(p, m, mask, eps)
    kl_qm = _kl(q, m, mask, eps)
    return 0.5 * (kl_pm + kl_qm) / math.log(2.0)


def compute_conflict(
    relevance: torch.Tensor,
    p_non_null: torch.Tensor,
    previous_belief: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute q_t^k = r_t^k * JS(p_tilde_{t,+}^k || b_{t-1}^k)."""
    return relevance * js_divergence(p_non_null, previous_belief, mask=mask, eps=eps)


def mask_unmatched_uncertainty(
    relevance: torch.Tensor,
    ambiguity: torch.Tensor,
    conflict: torch.Tensor,
    *,
    relevance_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask ambiguity/conflict for slots below the paper relevance threshold."""
    if relevance_threshold <= 0.0:
        matched = torch.ones_like(relevance, dtype=torch.bool)
        return matched, ambiguity, conflict
    matched = relevance >= float(relevance_threshold)
    zeros = torch.zeros_like(ambiguity)
    return matched, torch.where(matched, ambiguity, zeros), torch.where(matched, conflict, zeros)


def compute_previous_confidence(previous_belief: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-12) -> torch.Tensor:
    """Compute c_{t-1}^k = 1 - H(b_{t-1}^k) / log(|V_k|)."""
    return 1.0 - entropy_normalized(previous_belief, mask=mask, eps=eps)


def _kl(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor | None, eps: float) -> torch.Tensor:
    term = p.clamp_min(eps) * (p.clamp_min(eps).log() - q.clamp_min(eps).log())
    if mask is not None:
        term = term.masked_fill(~mask.bool(), 0.0)
    return term.sum(dim=-1)
