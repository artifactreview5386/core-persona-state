"""Slot-factorized belief and gated posterior update."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .evidence import NULL_INDEX
from .router import COMMIT_INDEX


DEFAULT_NEW_VALUE_PRIOR = 1e-6


@dataclass
class BeliefState:
    """Tensor slot-factorized belief over non-null values."""

    distribution: torch.Tensor
    mask: torch.Tensor | None = None


class BeliefUpdater(nn.Module):
    """Apply the CORE gated posterior update."""

    def forward(
        self,
        previous_belief: torch.Tensor,
        evidence_logits: torch.Tensor,
        gate: torch.Tensor,
        actions: torch.Tensor,
        value_mask: torch.Tensor | None = None,
        new_value_prior: float = DEFAULT_NEW_VALUE_PRIOR,
    ) -> torch.Tensor:
        return update_belief(
            previous_belief,
            evidence_logits,
            gate,
            actions,
            value_mask=value_mask,
            new_value_prior=new_value_prior,
        )


def update_belief(
    previous_belief: torch.Tensor,
    evidence_logits: torch.Tensor,
    gate: torch.Tensor,
    actions: torch.Tensor,
    value_mask: torch.Tensor | None = None,
    new_value_prior: float = DEFAULT_NEW_VALUE_PRIOR,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Update only COMMIT slots; DEFER and IGNORE preserve the prior exactly."""
    value_logits = evidence_logits[..., : evidence_logits.size(-1) + NULL_INDEX]
    extended_previous = extend_belief_values(
        previous_belief,
        target_values=value_logits.size(-1),
        new_value_prior=new_value_prior,
        eps=eps,
    )
    value_mask = _align_value_mask(value_mask, target_values=value_logits.size(-1))

    gated_logits = gate.unsqueeze(-1) * value_logits
    unnormalized = extended_previous.clamp_min(eps) * torch.exp(gated_logits)
    if value_mask is not None:
        unnormalized = unnormalized.masked_fill(~value_mask.bool(), 0.0)
    posterior = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(eps)

    commit_mask = actions.eq(COMMIT_INDEX).unsqueeze(-1)
    preserved_previous = preserve_belief_values(previous_belief, target_values=value_logits.size(-1))
    return torch.where(commit_mask, posterior, preserved_previous)


def extend_belief_values(
    previous_belief: torch.Tensor,
    *,
    target_values: int,
    new_value_prior: float = DEFAULT_NEW_VALUE_PRIOR,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Add paper-specified small prior mass when evidence introduces new values."""
    current_values = previous_belief.size(-1)
    if target_values < current_values:
        raise ValueError("evidence logits cannot have fewer non-null values than previous_belief")
    if target_values == current_values:
        return previous_belief
    extra_shape = (*previous_belief.shape[:-1], target_values - current_values)
    extra = previous_belief.new_full(extra_shape, float(new_value_prior))
    extended = torch.cat([previous_belief, extra], dim=-1)
    return extended / extended.sum(dim=-1, keepdim=True).clamp_min(eps)


def preserve_belief_values(previous_belief: torch.Tensor, *, target_values: int) -> torch.Tensor:
    """Pad without assigning probability mass when memory is not updated."""
    current_values = previous_belief.size(-1)
    if target_values < current_values:
        raise ValueError("target_values cannot be smaller than previous_belief")
    if target_values == current_values:
        return previous_belief
    extra_shape = (*previous_belief.shape[:-1], target_values - current_values)
    extra = previous_belief.new_zeros(extra_shape)
    return torch.cat([previous_belief, extra], dim=-1)


def _align_value_mask(value_mask: torch.Tensor | None, *, target_values: int) -> torch.Tensor | None:
    if value_mask is None:
        return None
    current_values = value_mask.size(-1)
    if current_values > target_values:
        raise ValueError("value_mask cannot have more values than evidence logits")
    if current_values == target_values:
        return value_mask
    extra_shape = (*value_mask.shape[:-1], target_values - current_values)
    extra = torch.ones(extra_shape, dtype=torch.bool, device=value_mask.device)
    return torch.cat([value_mask.bool(), extra], dim=-1)


def confidence_from_belief(previous_belief: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-12) -> torch.Tensor:
    """Entropy-normalized confidence c_t^k = 1 - H(b_t^k) / log(|V_k|)."""
    from .uncertainty import compute_previous_confidence

    return compute_previous_confidence(previous_belief, mask=mask, eps=eps)
