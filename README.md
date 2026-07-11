# CORE Anonymous Review Artifact

This repository contains the anonymous, reviewer-facing implementation of the
CORE method for long-term personalized dialogue with controlled persona belief
updates.

The code is organized around the paper method, not around a general RLHF
framework. The executable CORE step is:

```text
slot-conditioned evidence logits
-> evidence distribution
-> relevance / ambiguity / conflict features
-> Eq.40 linear softmax COMMIT / DEFER / IGNORE routing
-> COMMIT-masked revision gate
-> long-term belief update
-> RESPOND / CLARIFY discourse decision
```

## Quick Check

Install the lightweight runtime:

```bash
python -m pip install -r requirements.txt
```

Run the full offline verification suite:

```bash
python -m pytest tests/
bash scripts/verify_paper_faithful.sh
```

A dependency-light artifact check is also available:

```bash
python -m openrlhf.cli.core_preflight --check-paths
```

For Windows environments without Bash:

```bash
python -m train.sft --config configs/paper_default.yaml --dry_run
python -m train.calibrate_uncertainty --config configs/paper_default.yaml --dry_run
python -m train.ppo --config configs/paper_default.yaml --dry_run
```

## Repository Map

| Path | Purpose |
|---|---|
| `core/` | Paper method modules: evidence, uncertainty, routing, revision gate, belief update, discourse, and Algorithm 1 pipeline |
| `train/` | SFT losses, uncertainty calibration, reward computation, and saved-rollout PPO objective |
| `configs/` | Paper-facing default configuration and thresholds |
| `scripts/` | Reproduction, training, and code-paper alignment commands |
| `tests/` | Unit tests for paper-code alignment and expected edge cases |
| `examples/` | Small synthetic files for schema and pipeline smoke checks |
| `openrlhf/` | Lightweight artifact utilities for config and data validation |

## Paper Components

| Paper component | Implementation |
|---|---|
| Slot-conditioned evidence logits and distribution | `core/evidence.py::EvidenceScorer`, `compute_evidence_distribution` |
| Relevance / ambiguity / conflict | `core/evidence.py::compute_relevance`, `core/uncertainty.py` |
| Eq.40 three-way routing | `core/router.py::UpdateRouter` |
| Semi-open value handling | `core/belief.py::extend_belief_values` |
| COMMIT-masked revision gate | `core/revision.py::RevisionGate` |
| Persona belief update | `core/belief.py::update_belief` |
| Discourse decision | `core/discourse.py::DiscoursePolicy` |
| End-to-end step | `core/pipeline.py::run_core_step` |
| SFT warm-up losses | `train/sft.py::compute_sft_losses` |
| Reward and PPO objective | `train/reward.py`, `train/ppo.py` |

The full mapping is in `CODE_PAPER_ALIGNMENT.md`.

## Configuration

The default configuration is `configs/paper_default.yaml`. Reward weights and
thresholds are read from config rather than hidden in code:

```yaml
thresholds:
  relevance: 0.05

reward:
  action: 0.35
  state: 0.30
  answer: 0.25
  clarification: 0.10
  kl_beta: 0.05
```

## Reproduction Commands

Formal method reproduction requires paper-preprocessed tensorized inputs. See
`DATA_FORMAT.md` for the training JSONL fields.

```bash
TRAIN_FILE=path/to/sft_train.jsonl \
DEV_FILE=path/to/sft_dev.jsonl \
CALIBRATION_FILE=path/to/uncertainty_calibration.jsonl \
ROLLOUT_FILE=path/to/ppo_rollouts.jsonl \
OUT=outputs/reproduce_main \
bash scripts/reproduce_main.sh
```

Single-stage commands:

```bash
python -m train.sft --config configs/paper_default.yaml --train_file path/to/train.jsonl --dev_file path/to/dev.jsonl --output_dir outputs/sft
python -m train.calibrate_uncertainty --config configs/paper_default.yaml --calibration_file path/to/calibration.jsonl --output_dir outputs/uncertainty
python -m train.ppo --config configs/paper_default.yaml --rollout_file path/to/rollouts.jsonl --uncertainty_head outputs/uncertainty/uncertainty_head.pt --output_dir outputs/ppo
```

## Method Guarantees Checked by Tests

- Relevance is computed from the evidence distribution and is not fixed to 1.0.
- Ambiguity uses entropy over the non-null evidence distribution.
- Conflict uses distribution divergence against previous belief.
- Routing and discourse control receive the full evidence distribution, including null.
- Low-relevance unmatched slots are forced to `IGNORE`.
- `DEFER` and `IGNORE` do not write long-term belief.
- New values are introduced with the configured positive small prior.
- Discourse is decided after belief update by an explicit module.
- Reward rows must contain the formal paper reward components.
- Formal SFT and PPO rows are rejected when required fields are missing.

## Declared Boundary

`train/ppo.py` computes the paper reward and clipped PPO objective on saved
rollout rows. This artifact does not include private datasets, large-model
checkpoints, or generated experiment logs. The public code is intentionally
limited to the paper's executable CORE method and training objectives.

## Additional Reviewer Files

- `ARTIFACT_EVALUATION.md`: step-by-step commands for checking the artifact.
- `CODE_PAPER_ALIGNMENT.md`: paper component to code/test/config mapping.
- `DATA_FORMAT.md`: formal JSONL schemas expected by training.
