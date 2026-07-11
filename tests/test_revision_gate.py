import torch

from core.revision import RevisionGate
from core.router import COMMIT_INDEX, DEFER_INDEX, IGNORE_INDEX


def test_gate_has_learnable_params_and_action_mask():
    gate = RevisionGate()
    params = list(gate.parameters())

    assert params
    assert sum(p.numel() for p in params) == 5

    relevance = torch.ones(1, 3)
    ambiguity = torch.zeros(1, 3)
    conflict = torch.zeros(1, 3)
    confidence = torch.ones(1, 3)
    actions = torch.tensor([[COMMIT_INDEX, DEFER_INDEX, IGNORE_INDEX]])
    values = gate(relevance, ambiguity, conflict, confidence, actions)

    assert values[0, 0].item() > 0.0
    assert values[0, 1].item() == 0.0
    assert values[0, 2].item() == 0.0
