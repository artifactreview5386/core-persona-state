import torch

from core.belief import update_belief
from core.router import COMMIT_INDEX, DEFER_INDEX, IGNORE_INDEX


def test_commit_updates_belief_and_defer_ignore_preserve():
    prior = torch.tensor([[[0.5, 0.5], [0.7, 0.3], [0.2, 0.8]]])
    logits = torch.tensor([[[2.0, 0.0, -10.0], [0.0, 2.0, -10.0], [2.0, 0.0, -10.0]]])
    gate = torch.tensor([[1.0, 1.0, 1.0]])
    actions = torch.tensor([[COMMIT_INDEX, DEFER_INDEX, IGNORE_INDEX]])

    posterior = update_belief(prior, logits, gate, actions)

    assert not torch.allclose(posterior[:, 0], prior[:, 0])
    assert torch.allclose(posterior[:, 1], prior[:, 1])
    assert torch.allclose(posterior[:, 2], prior[:, 2])
    assert posterior.shape[-1] == 2
    assert torch.allclose(posterior.sum(dim=-1), torch.ones(1, 3), atol=1e-6)


def test_new_evidence_value_extends_belief_with_small_prior():
    prior = torch.tensor([[[0.6, 0.4]]])
    logits = torch.tensor([[[0.0, 0.0, 3.0, -10.0]]])
    gate = torch.tensor([[1.0]])
    actions = torch.tensor([[COMMIT_INDEX]])

    posterior = update_belief(prior, logits, gate, actions, new_value_prior=1e-3)

    assert posterior.shape[-1] == 3
    assert posterior[0, 0, 2] > 0.0
    assert torch.allclose(posterior.sum(dim=-1), torch.ones(1, 1), atol=1e-6)


def test_defer_ignore_do_not_assign_mass_to_new_values():
    prior = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
    logits = torch.tensor([[[0.0, 0.0, 4.0, -10.0], [0.0, 0.0, 4.0, -10.0]]])
    gate = torch.tensor([[1.0, 1.0]])
    actions = torch.tensor([[DEFER_INDEX, IGNORE_INDEX]])

    posterior = update_belief(prior, logits, gate, actions, new_value_prior=1e-3)

    assert posterior.shape[-1] == 3
    assert torch.allclose(posterior[..., :2], prior)
    assert torch.allclose(posterior[..., 2], torch.zeros(1, 2))
