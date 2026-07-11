import torch

from core.discourse import DISCOURSE_ACTIONS, DiscoursePolicy
from core.router import DEFER_INDEX


def test_discourse_logits_and_defer_not_automatic_clarify():
    policy = DiscoursePolicy(feature_dim=6, hidden_dim=8)
    out = policy(features=torch.randn(2, 6))

    assert DISCOURSE_ACTIONS == ("RESPOND", "CLARIFY")
    assert out.logits.shape == (2, 2)
    assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)

    policy_from_components = DiscoursePolicy(hidden_dim=8)
    update_actions = torch.full((2, 3), DEFER_INDEX)
    out_with_defer = policy_from_components(
        utterance_repr=torch.randn(2, 4),
        history_repr=torch.randn(2, 4),
        belief_repr=torch.randn(2, 3, 2),
        slot_features=torch.randn(2, 3, 4),
        update_actions=update_actions,
    )
    assert out_with_defer.logits.shape == (2, 2)
