import torch

from core.router import ACTION_SPACE, IGNORE_INDEX, RouterOutput, UpdateRouter, apply_relevance_threshold


def test_router_logits_shape_softmax_and_action_space():
    router = UpdateRouter(feature_dim=9)
    features = torch.randn(2, 4, 9)
    out = router(features=features)

    assert ACTION_SPACE == ("COMMIT", "DEFER", "IGNORE")
    assert isinstance(router.linear, torch.nn.Linear)
    assert router.linear.in_features == 9
    assert router.linear.out_features == len(ACTION_SPACE)
    assert out.logits.shape == (2, 4, 3)
    assert out.probabilities.shape == (2, 4, 3)
    assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    assert out.actions.shape == (2, 4)


def test_router_uses_continuous_relevance_features():
    router = UpdateRouter(feature_dim=4)
    features = torch.tensor([[[0.01, 0.80, 0.60, 0.50], [0.90, 0.70, 0.50, 0.41]]])

    out = router(features=features)

    assert out.probabilities.shape == (1, 2, 3)
    assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(1, 2), atol=1e-6)


def test_relevance_threshold_forces_unmatched_slots_to_ignore():
    raw = RouterOutput(
        logits=torch.zeros(1, 2, 3),
        probabilities=torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]]),
        actions=torch.tensor([[0, 1]]),
    )
    relevance = torch.tensor([[0.01, 0.20]])

    out = apply_relevance_threshold(raw, relevance, relevance_threshold=0.05)

    assert out.actions[0, 0].item() == IGNORE_INDEX
    assert out.probabilities[0, 0, IGNORE_INDEX].item() == 1.0
    assert out.actions[0, 1].item() == 1
