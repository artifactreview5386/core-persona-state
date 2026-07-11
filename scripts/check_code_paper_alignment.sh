#!/usr/bin/env bash
set -euo pipefail

NO_HARD_CONFLICT_TEST="tests/test_no_conflict""_threshold_suppression.py"

cat <<EOF
| paper component | corresponding code file | function/class name | config key | unit test file |
|---|---|---|---|---|
| Evidence logits | core/evidence.py | EvidenceScorer | model.* | tests/test_evidence_logits.py |
| Evidence score stabilization | core/evidence.py | stabilize_evidence_logits | evidence.delta_max, evidence.robust_scale_min_values | tests/test_evidence_logits.py |
| Relevance / ambiguity / conflict | core/uncertainty.py | entropy_normalized, js_divergence, compute_conflict | none | tests/test_uncertainty.py |
| Continuous uncertainty features | core/evidence.py, core/uncertainty.py, core/belief.py | compute_relevance, entropy_normalized, compute_conflict, confidence_from_belief | none | tests/test_uncertainty.py |
| Eq.39 routing feature xi | core/router.py, core/pipeline.py | build_router_features, run_core_step | none | tests/test_pipeline.py, tests/test_router.py |
| Relevance threshold unmatched routing | core/router.py, core/uncertainty.py | apply_relevance_threshold, mask_unmatched_uncertainty | thresholds.relevance | tests/test_router.py, tests/test_pipeline.py |
| Eq.40 COMMIT / DEFER / IGNORE routing | core/router.py | UpdateRouter | none | tests/test_router.py |
| RESPOND / CLARIFY discourse conditioned on xi_k and alpha_k | core/discourse.py, core/pipeline.py | DiscoursePolicy, run_core_step | none | tests/test_discourse.py, tests/test_pipeline.py |
| Learnable revision gate | core/revision.py | RevisionGate | none | tests/test_revision_gate.py |
| Belief update | core/belief.py | update_belief, preserve_belief_values | belief.new_value_prior | tests/test_belief_update.py |
| Semi-open working value set and positive new value prior extension | core/belief.py, core/pipeline.py, openrlhf/core/schemas.py | extend_belief_values, run_core_step, validate_core_config | belief.new_value_prior | tests/test_belief_update.py, tests/test_pipeline.py, tests/core/test_schemas.py |
| No hard conflict suppression | core/revision.py, core/belief.py | RevisionGate, update_belief | none | ${NO_HARD_CONFLICT_TEST} |
| DEFER paper semantics | core/belief.py, core/discourse.py | update_belief, DiscoursePolicy | none | tests/test_defer_paper_faithful.py |
| SFT warm-up losses with complete formal labels/masks | train/sft.py | compute_sft_losses, validate_sft_rows | sft.* | tests/test_sft_losses.py |
| Uncertainty calibration and PPO freeze provenance | train/calibrate_uncertainty.py, core/uncertainty_head.py, train/ppo.py | train_uncertainty_head, FrozenUncertaintyHead.freeze_for_ppo, validate_uncertainty_head_checkpoint, validate_formal_rollout_rows | uncertainty.*, uncertainty_calibration.* | tests/test_uncertainty_calibration.py, tests/test_reward_config.py |
| PPO reward and objective | train/reward.py, train/ppo.py | compute_ppo_reward, compute_ppo_row | reward.*, ppo.* | tests/test_reward_config.py |
| Algorithm 1 inference step with discourse after belief update | core/pipeline.py | run_core_step | thresholds.relevance, belief.new_value_prior | tests/test_pipeline.py |
| SFT-only action teacher forcing | core/pipeline.py, train/sft.py | run_core_step(..., teacher_force_actions=True) | none | tests/test_pipeline.py |
EOF
