"""Learnable sigmoid revision gate for paper-faithful CORE."""

from __future__ import annotations

import torch
from torch import nn

from .router import COMMIT_INDEX


class RevisionGate(nn.Module):
    """g_t^k = I[action == COMMIT] * sigmoid(w_g^T psi_t^k + b_g)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(
        self,
        relevance: torch.Tensor,
        ambiguity: torch.Tensor,
        conflict: torch.Tensor,
        previous_confidence: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        psi = torch.stack(
            [relevance, 1.0 - ambiguity, 1.0 - conflict, previous_confidence],
            dim=-1,
        )
        gate = torch.sigmoid(self.linear(psi)).squeeze(-1)
        commit_mask = actions.eq(COMMIT_INDEX).to(dtype=gate.dtype)
        return gate * commit_mask
