# Code-Paper Alignment

| Paper component | Code file | Function / class | Config key | Unit test |
|---|---|---|---|---|
| Slot-conditioned evidence logits | `core/evidence.py` | `EvidenceScorer` | `model.*` | `tests/test_evidence_logits.py` |
| Evidence distribution | `core/evidence.py` | `compute_evidence_distribution` | none | `tests/test_evidence_logits.py` |
| Evidence score stabilization | `core/evidence.py` | `stabilize_evidence_logits` | `evidence.delta_max`, `evidence.robust_scale_min_values` | `tests/test_evidence_logits.py` |
| Relevance | `core/evidence.py` | `compute_relevance` | none | `tests/test_evidence_logits.py` |
| Ambiguity | `core/uncertainty.py` | `entropy_normalized` | none | `tests/test_uncertainty.py` |
| Conflict | `core/uncertainty.py` | `compute_conflict`, `js_divergence` | none | `tests/test_uncertainty.py` |
| Previous belief confidence | `core/belief.py`, `core/uncertainty.py` | `confidence_from_belief`, `compute_previous_confidence` | none | `tests/test_uncertainty.py` |
| Eq.39 routing feature `xi` | `core/router.py`, `core/pipeline.py` | `build_router_features`, `run_core_step` | none | `tests/test_pipeline.py`, `tests/test_router.py` |
| Relevance-threshold unmatched slots | `core/router.py`, `core/uncertainty.py` | `apply_relevance_threshold`, `mask_unmatched_uncertainty` | `thresholds.relevance` | `tests/test_router.py`, `tests/test_pipeline.py` |
| Eq.40 COMMIT / DEFER / IGNORE routing | `core/router.py` | `UpdateRouter` | none | `tests/test_router.py` |
| Semi-open working value set | `core/belief.py`, `core/pipeline.py` | `extend_belief_values`, `run_core_step` | `belief.new_value_prior` | `tests/test_belief_update.py`, `tests/test_pipeline.py` |
| Revision gate | `core/revision.py` | `RevisionGate` | none | `tests/test_revision_gate.py` |
| Belief update | `core/belief.py` | `update_belief`, `preserve_belief_values` | `belief.new_value_prior` | `tests/test_belief_update.py` |
| Algorithm 1 executable path | `core/pipeline.py` | `run_core_step` | `thresholds.relevance`, `belief.new_value_prior` | `tests/test_pipeline.py` |
| RESPOND / CLARIFY discourse | `core/discourse.py`, `core/pipeline.py` | `DiscoursePolicy`, `run_core_step` | none | `tests/test_discourse.py`, `tests/test_pipeline.py` |
| SFT warm-up losses | `train/sft.py` | `compute_sft_losses`, `validate_sft_rows` | `sft.*` | `tests/test_sft_losses.py` |
| Uncertainty calibration and freeze | `train/calibrate_uncertainty.py`, `core/uncertainty_head.py` | `train_uncertainty_head`, `FrozenUncertaintyHead.freeze_for_ppo` | `uncertainty.*`, `uncertainty_calibration.*` | `tests/test_uncertainty_calibration.py` |
| PPO reward | `train/reward.py` | `compute_ppo_reward` | `reward.*` | `tests/test_reward_config.py` |
| PPO objective on saved rollouts | `train/ppo.py` | `compute_ppo_row`, `validate_formal_rollout_rows` | `ppo.*` | `tests/test_reward_config.py` |
| Reviewer preflight | `openrlhf/core/schemas.py` | `validate_core_config`, `normalize_dialogue_record` | `configs/*.yaml` | `tests/core/test_schemas.py` |
