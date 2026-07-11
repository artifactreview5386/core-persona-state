"""Evidence logits and evidence distribution for paper-faithful CORE."""

from __future__ import annotations

import torch
from torch import nn


NULL_INDEX = -1


class EvidenceScorer(nn.Module):
    """Slot-conditioned evidence scorer.

    Given current utterance/context representation h_t, dialogue history
    representation, slot representation e_k, value representations e_v, a
    previous belief summary b_{t-1}, and an explicit null option, this module
    returns raw local logits ell_t^k(v) over candidate values plus null. The
    executable pipeline stabilizes them into Delta_t^k(v) before softmax or
    belief revision.
    """

    def __init__(
        self,
        context_dim: int,
        slot_dim: int,
        value_dim: int,
        belief_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.history_proj = nn.Linear(context_dim, hidden_dim)
        self.slot_proj = nn.Linear(slot_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.belief_proj = nn.Linear(belief_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.null_value = nn.Parameter(torch.zeros(value_dim))

    def forward(
        self,
        context_repr: torch.Tensor,
        slot_repr: torch.Tensor,
        value_repr: torch.Tensor,
        previous_belief_summary: torch.Tensor,
        history_repr: torch.Tensor | None = None,
        null_option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return evidence logits with shape [batch, slots, values + null]."""
        if context_repr.dim() != 2:
            raise ValueError("context_repr must have shape [batch, context_dim]")
        batch = context_repr.size(0)
        if history_repr is None:
            history_repr = torch.zeros_like(context_repr)
        if history_repr.shape != context_repr.shape:
            raise ValueError("history_repr must match context_repr shape")

        if slot_repr.dim() == 2:
            slot_repr = slot_repr.unsqueeze(0).expand(batch, -1, -1)
        if slot_repr.dim() != 3:
            raise ValueError("slot_repr must have shape [slots, dim] or [batch, slots, dim]")

        if value_repr.dim() == 3:
            value_repr = value_repr.unsqueeze(0).expand(batch, -1, -1, -1)
        if value_repr.dim() != 4:
            raise ValueError("value_repr must have shape [slots, values, dim] or [batch, slots, values, dim]")

        slots, values = value_repr.size(1), value_repr.size(2)
        if slot_repr.size(1) != slots:
            raise ValueError("slot_repr and value_repr disagree on slot count")

        if previous_belief_summary.dim() == 2:
            previous_belief_summary = previous_belief_summary.unsqueeze(1).expand(-1, slots, -1)
        if previous_belief_summary.dim() != 3:
            raise ValueError("previous_belief_summary must have shape [batch, slots, belief_dim]")

        if null_option_repr is None:
            null_option_repr = self.null_value.view(1, 1, 1, -1)
        elif null_option_repr.dim() == 1:
            null_option_repr = null_option_repr.view(1, 1, 1, -1)
        elif null_option_repr.dim() == 2:
            null_option_repr = null_option_repr.view(1, slots, 1, -1)
        elif null_option_repr.dim() == 3:
            null_option_repr = null_option_repr.unsqueeze(2)
        if null_option_repr.size(-1) != value_repr.size(-1):
            raise ValueError("null_option_repr must have value_dim as the last dimension")
        null_repr = null_option_repr.expand(batch, slots, 1, -1)
        all_value_repr = torch.cat([value_repr, null_repr], dim=2)

        context_h = self.context_proj(context_repr).view(batch, 1, 1, -1)
        history_h = self.history_proj(history_repr).view(batch, 1, 1, -1)
        slot_h = self.slot_proj(slot_repr).view(batch, slots, 1, -1)
        value_h = self.value_proj(all_value_repr)
        belief_h = self.belief_proj(previous_belief_summary).view(batch, slots, 1, -1)
        hidden = torch.tanh(context_h + history_h + slot_h + value_h + belief_h)
        return self.scorer(hidden).squeeze(-1)


def stabilize_evidence_logits(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    delta_max: float | None = None,
    robust_scale_min_values: int | None = None,
    center: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply paper Appendix A evidence centering and clipping within each slot."""
    if logits.dim() != 3:
        raise ValueError("logits must have shape [batch, slots, values_plus_null]")
    stabilized = logits
    if center:
        if mask is None:
            mean = stabilized.mean(dim=-1, keepdim=True)
        else:
            if mask.shape != logits.shape:
                raise ValueError("mask must match logits shape")
            valid = mask.bool()
            denom = valid.sum(dim=-1, keepdim=True).to(dtype=logits.dtype).clamp_min(eps)
            mean = stabilized.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True) / denom
        stabilized = stabilized - mean
    if robust_scale_min_values is not None:
        if int(robust_scale_min_values) <= 0:
            raise ValueError("robust_scale_min_values must be positive when provided")
        scale = _robust_within_slot_scale(stabilized, mask, min_values=int(robust_scale_min_values), eps=eps)
        stabilized = stabilized / scale
    if delta_max is not None:
        if float(delta_max) <= 0.0:
            raise ValueError("delta_max must be positive when provided")
        stabilized = stabilized.clamp(min=-float(delta_max), max=float(delta_max))
    return stabilized


def compute_evidence_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    delta_max: float | None = None,
    robust_scale_min_values: int | None = None,
    center: bool = True,
) -> torch.Tensor:
    """Masked softmax over centered/clipped candidate values plus explicit null."""
    logits = stabilize_evidence_logits(
        logits,
        mask,
        delta_max=delta_max,
        robust_scale_min_values=robust_scale_min_values,
        center=center,
    )
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError("mask must match logits shape")
        logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1)


def compute_relevance(p_tilde: torch.Tensor, null_index: int = NULL_INDEX) -> torch.Tensor:
    """Compute r_t^k = 1 - p_tilde_t^k(null)."""
    return 1.0 - p_tilde.select(dim=-1, index=null_index)


def normalize_non_null_distribution(
    p_tilde: torch.Tensor,
    mask: torch.Tensor | None = None,
    null_index: int = NULL_INDEX,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Renormalize evidence probabilities after removing the null option."""
    non_null = _remove_null(p_tilde, null_index)
    if mask is not None:
        non_null_mask = _remove_null(mask.bool(), null_index)
        non_null = non_null.masked_fill(~non_null_mask, 0.0)
    denom = non_null.sum(dim=-1, keepdim=True).clamp_min(eps)
    return non_null / denom


def _remove_null(x: torch.Tensor, null_index: int) -> torch.Tensor:
    index = null_index if null_index >= 0 else x.size(-1) + null_index
    return torch.cat([x[..., :index], x[..., index + 1 :]], dim=-1)


def _robust_within_slot_scale(
    centered: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    min_values: int,
    eps: float,
) -> torch.Tensor:
    if mask is None:
        valid_count = torch.full_like(centered[..., :1], centered.size(-1), dtype=torch.long)
        scale = centered.abs().median(dim=-1, keepdim=True).values
    else:
        valid = mask.bool()
        valid_count = valid.sum(dim=-1, keepdim=True)
        abs_dev = centered.abs().masked_fill(~valid, float("nan"))
        scale = torch.nanmedian(abs_dev, dim=-1, keepdim=True).values
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    should_scale = valid_count >= int(min_values)
    return torch.where(should_scale, scale.clamp_min(eps), torch.ones_like(scale))
