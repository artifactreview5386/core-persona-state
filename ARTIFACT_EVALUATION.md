# Artifact Evaluation Guide

This document gives a short path for reviewers to inspect and run the anonymous
CORE artifact.

## 1. Environment

```bash
python -m pip install -r requirements.txt
```

The offline tests use `torch`, `pytest`, and `PyYAML`. No model checkpoint is
downloaded by the smoke checks.

## 2. Unit Tests

```bash
python -m pytest tests/
```

Expected result: all tests pass.

## 3. Paper-Faithful Verification

```bash
bash scripts/verify_paper_faithful.sh
```

This command runs the unit tests and scans the paper-facing code for forbidden
non-paper mechanisms.

## 4. Config and Dataset Preflight

```bash
python -m openrlhf.cli.validate_core_config --config configs/paper_default.yaml
python -m openrlhf.cli.core_preflight --check-paths
```

The preflight validates config structure, example dialogue schema, and
old-method-name absence.

## 5. Dry-Run Training

```bash
python -m train.sft --config configs/paper_default.yaml --dry_run
python -m train.calibrate_uncertainty --config configs/paper_default.yaml --dry_run
python -m train.ppo --config configs/paper_default.yaml --dry_run
```

These commands validate the wiring of each stage without requiring private data
or large model checkpoints.

## 6. Full Reproduction Inputs

Full reproduction requires:

- tensorized SFT train/dev JSONL rows,
- uncertainty calibration JSONL rows,
- saved PPO rollout rows with matching uncertainty-head fingerprints.

The required fields are listed in `DATA_FORMAT.md`.
