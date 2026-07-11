"""Executable CORE inference step matching the paper pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .belief import BeliefUpdater, extend_belief_values
from .discourse import DiscourseOutput, DiscoursePolicy
from .evidence import (
    EvidenceScorer,
    compute_evidence_distribution,
    compute_relevance,
    normalize_non_null_distribution,
    stabilize_evidence_logits,
)
from .revision import RevisionGate
from .router import RouterOutput, UpdateRouter, apply_relevance_threshold
from .uncertainty import compute_conflict, compute_previous_confidence, entropy_normalized, mask_unmatched_uncertainty


class COREModules(Protocol):
    evidence: EvidenceScorer
    router: UpdateRouter
    discourse: DiscoursePolicy
    revision_gate: RevisionGate
    belief_updater: BeliefUpdater


@dataclass
class COREStepOutput:
    evidence_logits: torch.Tensor
    evidence_distribution: torch.Tensor
    non_null_evidence: torch.Tensor
    relevance: torch.Tensor
    ambiguity: torch.Tensor
    conflict: torch.Tensor
    previous_confidence: torch.Tensor
    matched_slots: torch.Tensor
    router: RouterOutput
    discourse: DiscourseOutput
    gate: torch.Tensor
    belief: torch.Tensor


def run_core_step(
    batch: dict[str, torch.Tensor],
    modules: COREModules,
    *,
    relevance_threshold: float,
    new_value_prior: float,
    evidence_delta_max: float | None = None,
    evidence_robust_scale_min_values: int | None = None,
    sample_actions: bool = False,
    sample_discourse: bool = False,
    teacher_force_actions: bool = False,
) -> COREStepOutput:
    """Run evidence -> routing -> gated belief update -> discourse."""
    raw_evidence_logits = modules.evidence(
        batch["context_repr"],
        batch["slot_repr"],
        batch["value_repr"],
        batch["previous_belief_summary"],
        history_repr=batch.get("history_repr"),
        null_option_repr=batch.get("null_option_repr"),
    )
    value_mask = batch.get("value_mask")
    evidence_mask = _evidence_mask(value_mask)
    evidence_logits = stabilize_evidence_logits(
        raw_evidence_logits,
        evidence_mask,
        delta_max=evidence_delta_max,
        robust_scale_min_values=evidence_robust_scale_min_values,
    )
    evidence_distribution = compute_evidence_distribution(evidence_logits, evidence_mask, center=False)
    relevance = compute_relevance(evidence_distribution)
    non_null_evidence = normalize_non_null_distribution(evidence_distribution, evidence_mask)
    previous_belief = batch["previous_belief"]
    previous_belief_for_working_set = extend_belief_values(
        previous_belief,
        target_values=non_null_evidence.size(-1),
        new_value_prior=new_value_prior,
    )

    ambiguity = entropy_normalized(non_null_evidence, value_mask)
    conflict = compute_conflict(relevance, non_null_evidence, previous_belief_for_working_set, mask=value_mask)
    matched_slots, ambiguity, conflict = mask_unmatched_uncertainty(
        relevance,
        ambiguity,
        conflict,
        relevance_threshold=relevance_threshold,
    )
    previous_confidence = compute_previous_confidence(previous_belief_for_working_set, mask=value_mask)

    router_raw = modules.router(
        evidence_distribution=evidence_distribution,
        previous_belief=previous_belief_for_working_set,
        relevance=relevance,
        ambiguity=ambiguity,
        conflict=conflict,
        previous_confidence=previous_confidence,
        sample=sample_actions,
    )
    router = apply_relevance_threshold(
        router_raw,
        relevance,
        relevance_threshold=relevance_threshold,
    )
    actions_for_downstream = batch["gold_update_action"] if teacher_force_actions and "gold_update_action" in batch else router.actions
    gate = modules.revision_gate(relevance, ambiguity, conflict, previous_confidence, actions_for_downstream)
    belief = modules.belief_updater(
        previous_belief,
        evidence_logits,
        gate,
        actions_for_downstream,
        value_mask=value_mask,
        new_value_prior=new_value_prior,
    )
    slot_controller_state = torch.cat(
        [
            evidence_distribution,
            previous_belief_for_working_set,
            torch.stack([relevance, ambiguity, conflict, previous_confidence], dim=-1),
        ],
        dim=-1,
    )
    discourse = modules.discourse(
        utterance_repr=batch["context_repr"],
        history_repr=batch.get("history_repr", batch["context_repr"]),
        belief_repr=belief,
        slot_features=slot_controller_state,
        update_actions=actions_for_downstream,
        sample=sample_discourse,
    )
    return COREStepOutput(
        evidence_logits=evidence_logits,
        evidence_distribution=evidence_distribution,
        non_null_evidence=non_null_evidence,
        relevance=relevance,
        ambiguity=ambiguity,
        conflict=conflict,
        previous_confidence=previous_confidence,
        matched_slots=matched_slots,
        router=router,
        discourse=discourse,
        gate=gate,
        belief=belief,
    )


def _evidence_mask(value_mask: torch.Tensor | None) -> torch.Tensor | None:
    if value_mask is None:
        return None
    return torch.cat(
        [value_mask.bool(), torch.ones_like(value_mask[..., :1], dtype=torch.bool)],
        dim=-1,
    )
