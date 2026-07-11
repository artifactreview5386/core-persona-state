#!/usr/bin/env python3
"""Paper-faithful PPO reward composition for CORE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openrlhf.core.schemas import validate_core_config

PAPER_REWARD_KEYS = ("action", "state", "answer", "clarification")
FORMAL_REWARD_ROW_KEYS = (*PAPER_REWARD_KEYS, "kl")


def load_config(path: str, *, validate: bool = True) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if validate:
        validate_core_config(config, check_paths=False)
    return config


def compute_ppo_reward(
    *,
    action: float,
    state: float,
    answer: float,
    clarification: float,
    kl: float,
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    reward_cfg = config["reward"]
    missing = [key for key in (*PAPER_REWARD_KEYS, "kl_beta") if key not in reward_cfg]
    if missing:
        raise KeyError(f"Missing reward config keys: {', '.join(missing)}")
    components = {
        "action": float(action),
        "state": float(state),
        "answer": float(answer),
        "clarification": float(clarification),
        "kl": float(kl),
    }
    total = (
        float(reward_cfg["action"]) * components["action"]
        + float(reward_cfg["state"]) * components["state"]
        + float(reward_cfg["answer"]) * components["answer"]
        + float(reward_cfg["clarification"]) * components["clarification"]
        - float(reward_cfg["kl_beta"]) * components["kl"]
    )
    return total, components


def format_reward_log(total: float, components: dict[str, float]) -> dict[str, float]:
    return {
        "reward_action": components["action"],
        "reward_state": components["state"],
        "reward_answer": components["answer"],
        "reward_clarification": components["clarification"],
        "reward_kl": components["kl"],
        "reward_total": total,
    }


def validate_reward_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = [key for key in FORMAL_REWARD_ROW_KEYS if key not in row]
        if missing:
            raise ValueError(f"reward row {index} is missing required keys: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute paper-faithful CORE PPO reward")
    parser.add_argument("--config", default="configs/paper_default.yaml")
    parser.add_argument("--input", default=None, help="Optional JSONL with action/state/answer/clarification/kl")
    parser.add_argument("--output", default="outputs/reward/rewards.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.dry_run:
        rows = [{"action": 1.0, "state": 1.0, "answer": 1.0, "clarification": 0.0, "kl": 0.0}]
    else:
        if not args.input:
            parser.error("--input is required unless --dry_run is set")
        rows = [json.loads(line) for line in open(args.input, "r", encoding="utf-8") if line.strip()]
        validate_reward_rows(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            total, components = compute_ppo_reward(
                action=row["action"],
                state=row["state"],
                answer=row["answer"],
                clarification=row["clarification"],
                kl=row["kl"],
                config=config,
            )
            payload = format_reward_log(total, components)
            f.write(json.dumps(payload) + "\n")
            print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
