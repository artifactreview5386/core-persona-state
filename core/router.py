"""Eq.40 linear softmax update routing head for paper-faithful CORE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


ACTION_SPACE = ("COMMIT", "DEFER", "IGNORE")
COMMIT_INDEX = 0
DEFER_INDEX = 1
IGNORE_INDEX = 2


@dataclass
class RouterOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    actions: torch.Tensor


class UpdateRouter(nn.Module):
    """pi_upd(alpha | xi) = softmax(W_alpha xi + b_alpha)."""

    def __init__(self, feature_dim: int | None = None) -> None:
        super().__init__()
        self.linear = (
            nn.Linear(feature_dim, len(ACTION_SPACE))
            if feature_dim is not None
            else nn.LazyLinear(len(ACTION_SPACE))
        )

    def forward(
        self,
        *,
        features: torch.Tensor | None = None,
        evidence_distribution: torch.Tensor | None = None,
        previous_belief: torch.Tensor | None = None,
        relevance: torch.Tensor | None = None,
        ambiguity: torch.Tensor | None = None,
        conflict: torch.Tensor | None = None,
        previous_confidence: torch.Tensor | None = None,
        sample: bool = False,
    ) -> RouterOutput:
        if features is None:
            features = build_router_features(
                evidence_distribution=evidence_distribution,
                previous_belief=previous_belief,
                relevance=relevance,
                ambiguity=ambiguity,
                conflict=conflict,
                previous_confidence=previous_confidence,
            )
        logits = self.linear(features)
        probabilities = torch.softmax(logits, dim=-1)
        if sample:
            actions = torch.distributions.Categorical(probs=probabilities).sample()
        else:
            actions = probabilities.argmax(dim=-1)
        return RouterOutput(logits=logits, probabilities=probabilities, actions=actions)


def apply_relevance_threshold(
    router_output: RouterOutput,
    relevance: torch.Tensor,
    *,
    relevance_threshold: float,
) -> RouterOutput:
    """Force below-threshold slots to IGNORE as in Algorithm 1."""
    if relevance_threshold <= 0.0:
        return router_output
    unmatched = relevance < float(relevance_threshold)
    forced_actions = torch.where(
        unmatched,
        torch.full_like(router_output.actions, IGNORE_INDEX),
        router_output.actions,
    )
    ignore_probs = F.one_hot(
        torch.full_like(router_output.actions, IGNORE_INDEX),
        num_classes=len(ACTION_SPACE),
    ).to(dtype=router_output.probabilities.dtype, device=router_output.probabilities.device)
    forced_probs = torch.where(unmatched.unsqueeze(-1), ignore_probs, router_output.probabilities)
    return RouterOutput(logits=router_output.logits, probabilities=forced_probs, actions=forced_actions)


def build_router_features(
    *,
    evidence_distribution: torch.Tensor | None,
    previous_belief: torch.Tensor | None,
    relevance: torch.Tensor | None,
    ambiguity: torch.Tensor | None,
    conflict: torch.Tensor | None,
    previous_confidence: torch.Tensor | None,
) -> torch.Tensor:
    """Build xi_t^k = (p_tilde, b_prev, r, a, q, c_prev)."""
    parts = []
    for tensor in (evidence_distribution, previous_belief):
        if tensor is not None:
            parts.append(tensor)
    for tensor in (relevance, ambiguity, conflict, previous_confidence):
        if tensor is None:
            raise ValueError("relevance, ambiguity, conflict, and previous_confidence are required")
        parts.append(tensor.unsqueeze(-1))
    if not parts:
        raise ValueError("router features are empty")
    return torch.cat(parts, dim=-1)
