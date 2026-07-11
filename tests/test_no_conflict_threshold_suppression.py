from pathlib import Path

import torch

from core.belief import update_belief
from core.belief import confidence_from_belief
from core.evidence import compute_evidence_distribution, compute_relevance, normalize_non_null_distribution
from core.revision import RevisionGate
from core.router import COMMIT_INDEX, UpdateRouter
from core.uncertainty import compute_conflict, entropy_normalized


def test_conflicting_commit_can_move_belief_without_hard_threshold_suppression():
    previous_belief = torch.tensor([[[0.99, 0.01]]])
    evidence_logits = torch.tensor([[[-4.0, 8.0, -10.0]]])

    p_tilde = compute_evidence_distribution(evidence_logits)
    relevance = compute_relevance(p_tilde)
    p_non_null = normalize_non_null_distribution(p_tilde)
    ambiguity = entropy_normalized(p_non_null)
    conflict = compute_conflict(relevance, p_non_null, previous_belief)
    previous_confidence = confidence_from_belief(previous_belief)

    assert relevance.item() > 0.99
    assert ambiguity.item() < 0.01
    assert conflict.item() > 0.5

    router = UpdateRouter(feature_dim=4)
    _force_router_action(router, COMMIT_INDEX)
    router_out = router(features=torch.stack([relevance, ambiguity, conflict, previous_confidence], dim=-1))
    assert router_out.actions.item() == COMMIT_INDEX

    gate_module = RevisionGate()
    _force_large_commit_gate(gate_module)
    gate = gate_module(relevance, ambiguity, conflict, previous_confidence, router_out.actions)
    posterior = update_belief(previous_belief, evidence_logits, gate, router_out.actions)

    assert posterior[0, 0, 1] > previous_belief[0, 0, 1]
    assert posterior[0, 0, 1] > posterior[0, 0, 0]


def test_core_code_has_no_hard_conflict_suppression_constant():
    text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("core/router.py", "core/revision.py", "core/belief.py", "train/sft.py")
    )
    forbidden_conflict_name = "_".join(("conflict", "threshold"))
    suppressed_gate = "gate " + "*= " + "0.35"
    left_scaled = "* " + "0.35"
    right_scaled = "0.35" + " *"

    assert forbidden_conflict_name not in text
    assert suppressed_gate not in text
    assert left_scaled not in text
    assert right_scaled not in text


def _force_router_action(router: UpdateRouter, action_index: int) -> None:
    with torch.no_grad():
        for param in router.parameters():
            param.zero_()
        bias = router.linear.bias
        bias.fill_(-10.0)
        bias[action_index] = 10.0


def _force_large_commit_gate(gate_module: RevisionGate) -> None:
    with torch.no_grad():
        gate_module.linear.weight.zero_()
        gate_module.linear.bias.fill_(8.0)
