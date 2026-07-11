"""Independent RESPOND / CLARIFY discourse policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .router import ACTION_SPACE


DISCOURSE_ACTIONS = ("RESPOND", "CLARIFY")
RESPOND_INDEX = 0
CLARIFY_INDEX = 1


@dataclass
class DiscourseOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    decisions: torch.Tensor


class DiscoursePolicy(nn.Module):
    """Learnable discourse head over RESPOND / CLARIFY.

    This module does not equate DEFER with CLARIFY. Update actions are only
    input features; the decision is made by this head.
    """

    def __init__(self, feature_dim: int | None = None, hidden_dim: int = 128) -> None:
        super().__init__()
        first = nn.Linear(feature_dim, hidden_dim) if feature_dim is not None else nn.LazyLinear(hidden_dim)
        self.net = nn.Sequential(first, nn.Tanh(), nn.Linear(hidden_dim, len(DISCOURSE_ACTIONS)))

    def forward(
        self,
        *,
        features: torch.Tensor | None = None,
        utterance_repr: torch.Tensor | None = None,
        history_repr: torch.Tensor | None = None,
        belief_repr: torch.Tensor | None = None,
        slot_features: torch.Tensor | None = None,
        update_actions: torch.Tensor | None = None,
        sample: bool = False,
    ) -> DiscourseOutput:
        if features is None:
            features = build_discourse_features(
                utterance_repr=utterance_repr,
                history_repr=history_repr,
                belief_repr=belief_repr,
                slot_features=slot_features,
                update_actions=update_actions,
            )
        logits = self.net(features)
        probabilities = torch.softmax(logits, dim=-1)
        if sample:
            decisions = torch.distributions.Categorical(probs=probabilities).sample()
        else:
            decisions = probabilities.argmax(dim=-1)
        return DiscourseOutput(logits=logits, probabilities=probabilities, decisions=decisions)


def build_discourse_features(
    *,
    utterance_repr: torch.Tensor | None,
    history_repr: torch.Tensor | None,
    belief_repr: torch.Tensor | None,
    slot_features: torch.Tensor | None,
    update_actions: torch.Tensor | None,
) -> torch.Tensor:
    parts = []
    for tensor in (utterance_repr, history_repr, belief_repr):
        if tensor is not None:
            parts.append(_pool_if_slot_level(tensor))
    if slot_features is not None:
        parts.append(_pool_if_slot_level(slot_features))
    if update_actions is not None:
        parts.append(_pool_if_slot_level(_one_hot_update_actions(update_actions)))
    if not parts:
        raise ValueError("discourse features are empty")
    return torch.cat(parts, dim=-1)


def _pool_if_slot_level(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=1) if tensor.dim() == 3 else tensor


def _one_hot_update_actions(update_actions: torch.Tensor) -> torch.Tensor:
    if update_actions.dim() != 2:
        raise ValueError("update_actions must have shape [batch, slots]")
    return F.one_hot(update_actions.long(), num_classes=len(ACTION_SPACE)).to(dtype=torch.float32, device=update_actions.device)
