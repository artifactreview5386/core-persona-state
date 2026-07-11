import torch
import pytest

from core.uncertainty import (
    compute_conflict,
    compute_previous_confidence,
    entropy_normalized,
    js_divergence,
    mask_unmatched_uncertainty,
)


def test_ambiguity_entropy_normalized():
    certain = torch.tensor([[[1.0, 0.0]]])
    ambiguous = torch.tensor([[[0.5, 0.5]]])

    assert entropy_normalized(certain).item() < 1e-5
    assert torch.allclose(entropy_normalized(ambiguous), torch.ones(1, 1), atol=1e-5)


def test_js_divergence_and_conflict_are_distribution_based():
    p = torch.tensor([[[0.9, 0.1]]])
    q_same = torch.tensor([[[0.9, 0.1]]])
    q_opposite = torch.tensor([[[0.1, 0.9]]])

    assert js_divergence(p, q_same).item() < 1e-6
    assert js_divergence(p, q_opposite).item() > 0.3

    high = compute_conflict(torch.tensor([[1.0]]), p, q_opposite)
    low = compute_conflict(torch.tensor([[0.1]]), p, q_opposite)

    assert high.item() > low.item()
    assert torch.allclose(high, js_divergence(p, q_opposite))


def test_previous_confidence_is_entropy_complement():
    peaked = torch.tensor([[[0.99, 0.01]]])
    flat = torch.tensor([[[0.5, 0.5]]])

    assert compute_previous_confidence(peaked).item() > compute_previous_confidence(flat).item()
    assert compute_previous_confidence(flat).item() == pytest.approx(0.0, abs=1e-5)


def test_below_threshold_slots_mask_ambiguity_and_conflict():
    relevance = torch.tensor([[0.01, 0.20]])
    ambiguity = torch.tensor([[0.9, 0.8]])
    conflict = torch.tensor([[0.7, 0.6]])

    matched, masked_ambiguity, masked_conflict = mask_unmatched_uncertainty(
        relevance,
        ambiguity,
        conflict,
        relevance_threshold=0.05,
    )

    assert matched.tolist() == [[False, True]]
    assert torch.allclose(masked_ambiguity, torch.tensor([[0.0, 0.8]]))
    assert torch.allclose(masked_conflict, torch.tensor([[0.0, 0.6]]))
