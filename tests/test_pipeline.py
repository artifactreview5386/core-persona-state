import torch

from core.belief import BeliefUpdater
from core.discourse import DiscourseOutput
from core.pipeline import run_core_step
from core.revision import RevisionGate
from core.router import COMMIT_INDEX, IGNORE_INDEX, RouterOutput


def test_core_step_forces_low_relevance_to_ignore_and_preserves_belief():
    modules = _SyntheticModules()
    batch = {
        "context_repr": torch.zeros(1, 2),
        "history_repr": torch.zeros(1, 2),
        "slot_repr": torch.zeros(1, 1),
        "value_repr": torch.zeros(1, 2, 1),
        "previous_belief_summary": torch.zeros(1, 1, 1),
        "previous_belief": torch.tensor([[[0.6, 0.4]]]),
        "value_mask": torch.ones(1, 1, 2, dtype=torch.bool),
    }

    out = run_core_step(batch, modules, relevance_threshold=0.05, new_value_prior=1e-6)

    assert out.router.actions.item() == IGNORE_INDEX
    assert out.matched_slots.item() is False
    assert out.ambiguity.item() == 0.0
    assert out.conflict.item() == 0.0
    assert torch.allclose(out.belief, batch["previous_belief"])


def test_core_step_does_not_use_gold_actions_unless_teacher_forced():
    modules = _PredictedIgnoreModules()
    batch = {
        "context_repr": torch.zeros(1, 2),
        "history_repr": torch.zeros(1, 2),
        "slot_repr": torch.zeros(1, 1),
        "value_repr": torch.zeros(1, 2, 1),
        "previous_belief_summary": torch.zeros(1, 1, 1),
        "previous_belief": torch.tensor([[[0.5, 0.5]]]),
        "value_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "gold_update_action": torch.tensor([[COMMIT_INDEX]]),
    }

    predicted = run_core_step(batch, modules, relevance_threshold=0.0, new_value_prior=1e-6)
    teacher_forced = run_core_step(
        batch,
        modules,
        relevance_threshold=0.0,
        new_value_prior=1e-6,
        teacher_force_actions=True,
    )

    assert predicted.router.actions.item() == IGNORE_INDEX
    assert torch.allclose(predicted.belief, batch["previous_belief"])
    assert not torch.allclose(teacher_forced.belief, batch["previous_belief"])


def test_core_step_passes_paper_alpha_actions_to_discourse():
    discourse = _RecordingDiscourse()
    modules = _PredictedIgnoreModules()
    modules.discourse = discourse
    batch = {
        "context_repr": torch.zeros(1, 2),
        "history_repr": torch.zeros(1, 2),
        "slot_repr": torch.zeros(1, 1),
        "value_repr": torch.zeros(1, 2, 1),
        "previous_belief_summary": torch.zeros(1, 1, 1),
        "previous_belief": torch.tensor([[[0.5, 0.5]]]),
        "value_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "gold_update_action": torch.tensor([[COMMIT_INDEX]]),
    }

    run_core_step(batch, modules, relevance_threshold=0.0, new_value_prior=1e-6)
    predicted_actions = discourse.last_update_actions.clone()
    predicted_slot_state_width = discourse.last_slot_features.size(-1)
    run_core_step(
        batch,
        modules,
        relevance_threshold=0.0,
        new_value_prior=1e-6,
        teacher_force_actions=True,
    )

    assert predicted_actions.item() == IGNORE_INDEX
    assert discourse.last_update_actions.item() == COMMIT_INDEX
    assert predicted_slot_state_width == 9


def test_core_step_extends_previous_belief_for_new_working_values_before_routing():
    discourse = _RecordingDiscourse()
    modules = _NewValueCommitModules()
    modules.discourse = discourse
    batch = {
        "context_repr": torch.zeros(1, 2),
        "history_repr": torch.zeros(1, 2),
        "slot_repr": torch.zeros(1, 1),
        "value_repr": torch.zeros(1, 3, 1),
        "previous_belief_summary": torch.zeros(1, 1, 1),
        "previous_belief": torch.tensor([[[0.7, 0.3]]]),
        "value_mask": torch.ones(1, 1, 3, dtype=torch.bool),
    }

    out = run_core_step(batch, modules, relevance_threshold=0.0, new_value_prior=1e-3)

    assert out.belief.shape[-1] == 3
    assert discourse.last_slot_features.shape[-1] == 11
    assert discourse.last_slot_features[0, 0, 6].item() > 0.0
    assert out.conflict.isfinite().all()


def test_core_step_passes_full_p_tilde_to_router():
    router = _RecordingRouter()
    modules = _PredictedIgnoreModules()
    modules.router = router
    batch = {
        "context_repr": torch.zeros(1, 2),
        "history_repr": torch.zeros(1, 2),
        "slot_repr": torch.zeros(1, 1),
        "value_repr": torch.zeros(1, 2, 1),
        "previous_belief_summary": torch.zeros(1, 1, 1),
        "previous_belief": torch.tensor([[[0.5, 0.5]]]),
        "value_mask": torch.ones(1, 1, 2, dtype=torch.bool),
    }

    out = run_core_step(batch, modules, relevance_threshold=0.0, new_value_prior=1e-6)

    assert router.last_evidence_distribution.shape[-1] == 3
    assert torch.allclose(router.last_evidence_distribution, out.evidence_distribution)
    assert torch.allclose(router.last_evidence_distribution.sum(dim=-1), torch.ones(1, 1))


def test_core_step_discourse_uses_updated_belief_after_commit():
    discourse = _RecordingDiscourse()
    modules = _PredictedCommitModules()
    modules.discourse = discourse
    batch = {
        "context_repr": torch.zeros(1, 2),
        "history_repr": torch.zeros(1, 2),
        "slot_repr": torch.zeros(1, 1),
        "value_repr": torch.zeros(1, 2, 1),
        "previous_belief_summary": torch.zeros(1, 1, 1),
        "previous_belief": torch.tensor([[[0.9, 0.1]]]),
        "value_mask": torch.ones(1, 1, 2, dtype=torch.bool),
    }

    out = run_core_step(batch, modules, relevance_threshold=0.0, new_value_prior=1e-6)

    assert torch.allclose(discourse.last_belief_repr, out.belief)
    assert not torch.allclose(discourse.last_belief_repr, batch["previous_belief"])


class _LowRelevanceEvidence:
    def __call__(self, *args, **kwargs):
        return torch.tensor([[[0.0, 0.0, 8.0]]])


class _CommitRouter:
    def __call__(self, **kwargs):
        return RouterOutput(
            logits=torch.tensor([[[8.0, 0.0, 0.0]]]),
            probabilities=torch.tensor([[[1.0, 0.0, 0.0]]]),
            actions=torch.tensor([[COMMIT_INDEX]]),
        )


class _RespondDiscourse:
    def __call__(self, **kwargs):
        return DiscourseOutput(
            logits=torch.tensor([[1.0, 0.0]]),
            probabilities=torch.tensor([[1.0, 0.0]]),
            decisions=torch.tensor([0]),
        )


class _RecordingDiscourse(_RespondDiscourse):
    def __init__(self):
        self.last_update_actions = None
        self.last_slot_features = None
        self.last_belief_repr = None

    def __call__(self, **kwargs):
        self.last_update_actions = kwargs["update_actions"].detach().clone()
        self.last_slot_features = kwargs["slot_features"].detach().clone()
        self.last_belief_repr = kwargs["belief_repr"].detach().clone()
        return super().__call__(**kwargs)


class _SyntheticModules:
    evidence = _LowRelevanceEvidence()
    router = _CommitRouter()
    discourse = _RespondDiscourse()
    revision_gate = RevisionGate()
    belief_updater = BeliefUpdater()


class _ConfidentEvidence:
    def __call__(self, *args, **kwargs):
        return torch.tensor([[[4.0, 0.0, -4.0]]])


class _IgnoreRouter:
    def __call__(self, **kwargs):
        return RouterOutput(
            logits=torch.tensor([[[0.0, 0.0, 8.0]]]),
            probabilities=torch.tensor([[[0.0, 0.0, 1.0]]]),
            actions=torch.tensor([[IGNORE_INDEX]]),
        )


class _RecordingRouter(_IgnoreRouter):
    def __init__(self):
        self.last_evidence_distribution = None
        self.last_previous_belief = None

    def __call__(self, **kwargs):
        self.last_evidence_distribution = kwargs["evidence_distribution"].detach().clone()
        self.last_previous_belief = kwargs["previous_belief"].detach().clone()
        return super().__call__(**kwargs)


class _ActionMaskedGate:
    def __call__(self, relevance, ambiguity, conflict, previous_confidence, actions):
        return actions.eq(COMMIT_INDEX).to(dtype=relevance.dtype)


class _PredictedIgnoreModules:
    evidence = _ConfidentEvidence()
    router = _IgnoreRouter()
    discourse = _RespondDiscourse()
    revision_gate = _ActionMaskedGate()
    belief_updater = BeliefUpdater()


class _NewValueEvidence:
    def __call__(self, *args, **kwargs):
        return torch.tensor([[[0.0, 0.0, 5.0, -5.0]]])


class _NewValueCommitModules:
    evidence = _NewValueEvidence()
    router = _CommitRouter()
    discourse = _RespondDiscourse()
    revision_gate = _ActionMaskedGate()
    belief_updater = BeliefUpdater()


class _PredictedCommitModules:
    evidence = _ConfidentEvidence()
    router = _CommitRouter()
    discourse = _RespondDiscourse()
    revision_gate = _ActionMaskedGate()
    belief_updater = BeliefUpdater()
