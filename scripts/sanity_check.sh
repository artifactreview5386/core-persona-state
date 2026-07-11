#!/usr/bin/env bash
set -euo pipefail

python -m pytest tests/
python -m train.sft --config configs/paper_default.yaml --dry_run
python -m train.calibrate_uncertainty --config configs/paper_default.yaml --dry_run
python -m train.ppo --config configs/paper_default.yaml --dry_run
