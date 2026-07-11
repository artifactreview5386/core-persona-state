#!/usr/bin/env python3
"""PPO objective and paper reward computation for tensorized CORE rollouts.

This entry point computes the paper-specific reward and clipped PPO objective
on saved rollout rows, which is the auditable release path for CORE-specific
training signals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from train.reward import compute_ppo_reward, format_reward_log, load_config


FORMAL_ROLLOUT_REQUIRED_KEYS = (
    "action",
    "state",
    "answer",
    "clarification",
    "kl",
    "old_logprob",
    "new_logprob",
    "advantage",
    "value",
    "return",
    "entropy",
    "uncertainty_head_fingerprint",
)


def compute_ppo_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    """Compute Eq.20 reward plus the clipped PPO surrogate for one rollout row."""
    reward, components = compute_ppo_reward(
        action=float(row["action"]),
        state=float(row["state"]),
        answer=float(row["answer"]),
        clarification=float(row["clarification"]),
        kl=float(row["kl"]),
        config=config,
    )
    ppo_cfg = config.get("ppo", {})
    reward_cfg = config.get("reward", {})
    clip_range = float(ppo_cfg["clip_range"])
    value_coef = float(ppo_cfg["value_coef"])
    entropy_coef = float(ppo_cfg["entropy_coef"])
    kl_beta = float(reward_cfg["kl_beta"])

    new_logprob = float(row["new_logprob"])
    old_logprob = float(row["old_logprob"])
    advantage = float(row["advantage"])
    ratio = math.exp(new_logprob - old_logprob)
    clipped_ratio = min(max(ratio, 1.0 - clip_range), 1.0 + clip_range)
    policy_objective = min(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -policy_objective

    value_loss = (float(row["value"]) - float(row["return"])) ** 2
    entropy = float(row["entropy"])
    kl_loss = kl_beta * float(row["kl"])
    total_loss = policy_loss + kl_loss + value_coef * value_loss - entropy_coef * entropy

    return {
        **format_reward_log(reward, components),
        "ppo_ratio": ratio,
        "ppo_clipped_ratio": clipped_ratio,
        "ppo_advantage": advantage,
        "ppo_policy_loss": policy_loss,
        "ppo_value_loss": value_loss,
        "ppo_kl_loss": kl_loss,
        "ppo_entropy": entropy,
        "ppo_total_loss": total_loss,
    }


def summarize_ppo_rows(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    data = list(rows)
    if not data:
        raise ValueError("cannot summarize zero PPO rows")
    keys = sorted(data[0])
    return {key: sum(row[key] for row in data) / len(data) for key in keys}


def load_rollout_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"no PPO rollout rows found in {path}")
    return rows


def validate_formal_rollout_rows(rows: list[dict[str, Any]], *, uncertainty_head_fingerprint: str) -> None:
    for index, row in enumerate(rows, start=1):
        missing = [key for key in FORMAL_ROLLOUT_REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"PPO rollout row {index} is missing required keys: {', '.join(missing)}")
        if str(row["uncertainty_head_fingerprint"]) != uncertainty_head_fingerprint:
            raise ValueError(
                f"PPO rollout row {index} uncertainty_head_fingerprint does not match the frozen checkpoint"
            )


def dry_run_rows() -> list[dict[str, float]]:
    return [
        {
            "action": 1.0,
            "state": 1.0,
            "answer": 1.0,
            "clarification": 0.0,
            "kl": 0.0,
            "old_logprob": -1.2,
            "new_logprob": -1.1,
            "advantage": 1.0,
            "value": 0.4,
            "return": 1.0,
            "entropy": 0.2,
        }
    ]


def validate_uncertainty_head_checkpoint(path: str) -> dict[str, Any]:
    import torch

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"uncertainty head checkpoint does not exist: {path}")
    checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"invalid uncertainty head checkpoint: {path}")
    if checkpoint.get("frozen_for_ppo") is not True:
        raise ValueError(f"uncertainty head checkpoint is not marked frozen_for_ppo: {path}")
    fingerprint = checkpoint.get("uncertainty_head_fingerprint") or state_dict_fingerprint(checkpoint["state_dict"])
    return {"fingerprint": fingerprint}


def state_dict_fingerprint(state_dict: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise ValueError(f"uncertainty head state_dict value is not a tensor: {key}")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CORE PPO objective/reward computation")
    parser.add_argument("--config", default="configs/paper_default.yaml")
    parser.add_argument("--rollout_file", default=None, help="JSONL rows with reward components and PPO logprobs")
    parser.add_argument("--uncertainty_head", default=None, help="Frozen uncertainty head checkpoint from calibration")
    parser.add_argument("--output_dir", default="outputs/ppo")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.dry_run:
        rows = dry_run_rows()
    else:
        if not args.rollout_file:
            parser.error("--rollout_file is required unless --dry_run is set")
        if not args.uncertainty_head:
            parser.error("--uncertainty_head is required unless --dry_run is set")
        uncertainty_head = validate_uncertainty_head_checkpoint(args.uncertainty_head)
        rows = load_rollout_rows(args.rollout_file)
        validate_formal_rollout_rows(rows, uncertainty_head_fingerprint=uncertainty_head["fingerprint"])

    computed = [compute_ppo_row(row, config) for row in rows]
    summary = summarize_ppo_rows(computed)
    summary["dry_run"] = bool(args.dry_run)
    summary["rollout_file"] = args.rollout_file
    summary["uncertainty_head"] = args.uncertainty_head
    if not args.dry_run:
        summary["uncertainty_head_fingerprint"] = uncertainty_head["fingerprint"]
    summary["num_rollouts"] = len(rows)
    summary["scope"] = "ppo_objective_on_saved_rollouts"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ppo_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in computed:
            f.write(json.dumps(row) + "\n")
    (output_dir / "ppo_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
