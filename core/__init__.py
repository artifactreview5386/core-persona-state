"""Paper-faithful CORE modules.

Default pipeline:
evidence logits -> uncertainty features -> softmax routing head -> discourse
head -> sigmoid revision gate -> gated belief update.
"""

from .belief import BeliefState, BeliefUpdater, confidence_from_belief, update_belief
from .discourse import DISCOURSE_ACTIONS, DiscourseOutput, DiscoursePolicy
from .evidence import (
    EvidenceScorer,
    compute_evidence_distribution,
    compute_relevance,
    normalize_non_null_distribution,
    stabilize_evidence_logits,
)
from .pipeline import COREStepOutput, run_core_step
from .revision import RevisionGate
from .router import ACTION_SPACE, RouterOutput, UpdateRouter, apply_relevance_threshold
from .uncertainty import compute_conflict, compute_previous_confidence, entropy_normalized, js_divergence, mask_unmatched_uncertainty
from .uncertainty_head import FrozenUncertaintyHead, UncertaintyHeadOutput

__all__ = [
    "ACTION_SPACE",
    "DISCOURSE_ACTIONS",
    "BeliefState",
    "BeliefUpdater",
    "DiscourseOutput",
    "DiscoursePolicy",
    "EvidenceScorer",
    "FrozenUncertaintyHead",
    "RevisionGate",
    "RouterOutput",
    "COREStepOutput",
    "UpdateRouter",
    "UncertaintyHeadOutput",
    "apply_relevance_threshold",
    "compute_conflict",
    "compute_evidence_distribution",
    "compute_previous_confidence",
    "compute_relevance",
    "confidence_from_belief",
    "entropy_normalized",
    "js_divergence",
    "mask_unmatched_uncertainty",
    "normalize_non_null_distribution",
    "run_core_step",
    "stabilize_evidence_logits",
    "update_belief",
]
