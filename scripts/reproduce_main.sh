#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/paper_default.yaml}"
OUT="${OUT:-outputs/reproduce_main}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" == "1" ]]; then
  python -m train.sft --config "$CONFIG" --dry_run --output_dir "$OUT/sft"
  python -m train.calibrate_uncertainty --config "$CONFIG" --dry_run --output_dir "$OUT/uncertainty"
  python -m train.ppo --config "$CONFIG" --dry_run --output_dir "$OUT/ppo"
  exit 0
fi

: "${TRAIN_FILE:?Set TRAIN_FILE to the tensorized SFT training JSONL.}"
: "${CALIBRATION_FILE:?Set CALIBRATION_FILE to the uncertainty calibration JSONL.}"
: "${ROLLOUT_FILE:?Set ROLLOUT_FILE to saved PPO rollout JSONL rows.}"

SFT_ARGS=(--config "$CONFIG" --train_file "$TRAIN_FILE" --output_dir "$OUT/sft")
if [[ -n "${DEV_FILE:-}" ]]; then
  SFT_ARGS+=(--dev_file "$DEV_FILE")
fi
if [[ -n "${MODEL_NAME_OR_PATH:-}" ]]; then
  SFT_ARGS+=(--model_name_or_path "$MODEL_NAME_OR_PATH")
fi

python -m train.sft "${SFT_ARGS[@]}"
python -m train.calibrate_uncertainty --config "$CONFIG" --calibration_file "$CALIBRATION_FILE" --output_dir "$OUT/uncertainty"
python -m train.ppo \
  --config "$CONFIG" \
  --rollout_file "$ROLLOUT_FILE" \
  --uncertainty_head "$OUT/uncertainty/uncertainty_head.pt" \
  --output_dir "$OUT/ppo"
