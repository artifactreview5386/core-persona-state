"""Frozen uncertainty pathway for Eq.58-59."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class UncertaintyHeadOutput:
    features: torch.Tensor
    relevance: torch.Tensor
    ambiguity: torch.Tensor
    conflict: torch.Tensor


class FrozenUncertaintyHead(nn.Module):
    """Implements Eq.58-59 as a calibratable pathway frozen during PPO.

    Eq.58: x_t,unc^k = f_unc-feat^frozen(H_{t-1}, u_t, b_{t-1}, k)
    Eq.59: (r_hat, a_hat, q_hat) = f_unc^frozen(x_t,unc^k)

    The head may be trained/calibrated during supervised warm-up or calibration,
    then `freeze_for_ppo()` must be called before PPO.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.uncertainty_head = nn.Linear(hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> UncertaintyHeadOutput:
        x_unc = self.feature_net(features)
        values = torch.sigmoid(self.uncertainty_head(x_unc))
        return UncertaintyHeadOutput(
            features=x_unc,
            relevance=values[..., 0],
            ambiguity=values[..., 1],
            conflict=values[..., 2],
        )

    def freeze_for_ppo(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def unfreeze_for_warmup(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)
        self.train()
