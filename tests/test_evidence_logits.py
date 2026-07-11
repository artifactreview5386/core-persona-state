import torch

from core.evidence import (
    EvidenceScorer,
    compute_evidence_distribution,
    compute_relevance,
    normalize_non_null_distribution,
    stabilize_evidence_logits,
)


def test_evidence_logits_shape_and_null_option():
    scorer = EvidenceScorer(context_dim=5, slot_dim=4, value_dim=3, belief_dim=2, hidden_dim=7)
    logits = scorer(
        torch.randn(2, 5),
        torch.randn(3, 4),
        torch.randn(3, 4, 3),
        torch.randn(2, 3, 2),
        history_repr=torch.randn(2, 5),
        null_option_repr=torch.randn(3, 3),
    )

    assert logits.shape == (2, 3, 5)


def test_masked_distribution_and_relevance_changes_with_null_logit():
    logits = torch.tensor([[[0.0, 0.0, 0.0]]])
    probs = compute_evidence_distribution(logits)
    relevance_a = compute_relevance(probs)

    logits_high_null = torch.tensor([[[0.0, 0.0, 5.0]]])
    probs_high_null = compute_evidence_distribution(logits_high_null)
    relevance_b = compute_relevance(probs_high_null)

    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 1))
    assert relevance_b.item() < relevance_a.item()
    assert normalize_non_null_distribution(probs).shape == (1, 1, 2)


def test_evidence_logits_are_centered_and_clipped_before_distribution():
    logits = torch.tensor([[[2.0, 4.0, 6.0]]])
    stabilized = stabilize_evidence_logits(logits, delta_max=10.0)

    assert torch.allclose(stabilized, torch.tensor([[[-2.0, 0.0, 2.0]]]))

    clipped = stabilize_evidence_logits(torch.tensor([[[-100.0, 0.0, 100.0]]]), delta_max=3.0)
    assert clipped.min().item() >= -3.0
    assert clipped.max().item() <= 3.0


def test_evidence_logits_use_robust_scale_when_candidate_set_is_large_enough():
    logits = torch.tensor([[[0.0, 2.0, 4.0, 6.0, 8.0]]])
    stabilized = stabilize_evidence_logits(logits, delta_max=10.0, robust_scale_min_values=4)

    assert torch.allclose(stabilized, torch.tensor([[[-2.0, -1.0, 0.0, 1.0, 2.0]]]))


def test_small_candidate_sets_skip_robust_scaling():
    logits = torch.tensor([[[0.0, 10.0, 20.0]]])
    stabilized = stabilize_evidence_logits(logits, delta_max=20.0, robust_scale_min_values=4)

    assert torch.allclose(stabilized, torch.tensor([[[-10.0, 0.0, 10.0]]]))


def test_evidence_stabilization_respects_candidate_mask():
    logits = torch.tensor([[[2.0, 4.0, 100.0]]])
    mask = torch.tensor([[[True, True, False]]])
    stabilized = stabilize_evidence_logits(logits, mask=mask, delta_max=10.0)
    probs = compute_evidence_distribution(logits, mask=mask, delta_max=10.0)

    assert torch.allclose(stabilized[..., :2], torch.tensor([[[-1.0, 1.0]]]))
    assert probs[..., -1].item() == 0.0
