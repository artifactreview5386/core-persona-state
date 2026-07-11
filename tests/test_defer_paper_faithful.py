from pathlib import Path

import torch

from core.belief import update_belief
from core.discourse import CLARIFY_INDEX, RESPOND_INDEX, DiscoursePolicy
from core.router import DEFER_INDEX


def test_defer_preserves_long_term_belief_exactly():
    previous_belief = torch.tensor([[[0.8, 0.2]]])
    evidence_logits = torch.tensor([[[0.0, 6.0, -8.0]]])
    gate = torch.tensor([[1.0]])
    actions = torch.tensor([[DEFER_INDEX]])

    posterior = update_belief(previous_belief, evidence_logits, gate, actions)

    assert torch.allclose(posterior, previous_belief)


def test_defer_does_not_commit_new_value_mass():
    previous_belief = torch.tensor([[[0.8, 0.2]]])
    evidence_logits = torch.tensor([[[0.0, 0.0, 6.0, -8.0]]])
    gate = torch.tensor([[1.0]])
    actions = torch.tensor([[DEFER_INDEX]])

    posterior = update_belief(previous_belief, evidence_logits, gate, actions)

    assert torch.allclose(posterior[..., :2], previous_belief)
    assert posterior[0, 0, 2].item() == 0.0


def test_defer_has_no_buffer_or_accumulation_path_in_core_code():
    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in Path("core").glob("*.py"))
    forbidden_buffer_name = "_".join(("evidence", "buffer"))
    forbidden_deferred_name = "_".join(("deferred", "evidence"))

    assert forbidden_buffer_name not in text
    assert forbidden_deferred_name not in text
    assert "accumulation" not in text
    assert "graduate" not in text


def test_defer_does_not_automatically_trigger_clarify():
    policy = DiscoursePolicy(feature_dim=13, hidden_dim=4)
    utterance_repr = torch.zeros(1, 2)
    history_repr = torch.zeros(1, 2)
    belief_repr = torch.zeros(1, 1, 2)
    slot_features = torch.zeros(1, 1, 4)
    update_actions = torch.tensor([[DEFER_INDEX]])

    _force_discourse_decision(policy, RESPOND_INDEX)
    respond_out = policy(
        utterance_repr=utterance_repr,
        history_repr=history_repr,
        belief_repr=belief_repr,
        slot_features=slot_features,
        update_actions=update_actions,
    )

    _force_discourse_decision(policy, CLARIFY_INDEX)
    clarify_out = policy(
        utterance_repr=utterance_repr,
        history_repr=history_repr,
        belief_repr=belief_repr,
        slot_features=slot_features,
        update_actions=update_actions,
    )

    assert respond_out.decisions.item() == RESPOND_INDEX
    assert clarify_out.decisions.item() == CLARIFY_INDEX


def _force_discourse_decision(policy: DiscoursePolicy, decision_index: int) -> None:
    with torch.no_grad():
        for param in policy.parameters():
            param.zero_()
        bias = policy.net[-1].bias
        bias.fill_(-5.0)
        bias[decision_index] = 5.0
