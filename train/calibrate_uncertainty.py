#!/usr/bin/env python3
"""Calibrate and freeze the CORE uncertainty pathway for PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from core.uncertainty_head import FrozenUncertaintyHead
from train.reward import load_config


def load_calibration_jsonl(path: str) -> list[dict[str, torch.Tensor]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                rows.append(
                    {
                        "features": torch.tensor(payload["features"], dtype=torch.float32),
                        "targets": _target_tensor(payload),
                    }
                )
    if not rows:
        raise ValueError(f"no uncertainty calibration rows found in {path}")
    return rows


def train_uncertainty_head(
    rows: list[dict[str, torch.Tensor]],
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[FrozenUncertaintyHead, dict[str, float]]:
    first = rows[0]["features"]
    input_dim = int(first.numel())
    hidden_dim = int(config["uncertainty"]["hidden_dim"])
    head = FrozenUncertaintyHead(input_dim=input_dim, hidden_dim=hidden_dim)
    cfg = config["uncertainty_calibration"]
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    learning_rate = float(cfg["learning_rate"])
    rng = random.Random(seed)
    totals = {"loss_unc": 0.0}
    steps = 0

    for _epoch in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_rows = [rows[index] for index in order[start : start + batch_size]]
            features = torch.stack([row["features"].reshape(-1) for row in batch_rows], dim=0)
            targets = torch.stack([row["targets"] for row in batch_rows], dim=0)
            output = head(features)
            pred = torch.stack([output.relevance, output.ambiguity, output.conflict], dim=-1)
            loss = F.mse_loss(pred, targets)
            _zero_grad(head)
            loss.backward()
            _sgd_step(head, learning_rate)
            totals["loss_unc"] += float(loss.detach().cpu())
            steps += 1

    if steps == 0:
        raise ValueError("uncertainty calibration produced zero steps")
    head.freeze_for_ppo()
    return head, {"loss_unc": totals["loss_unc"] / steps, "calibration_steps": float(steps)}


def dry_run_rows() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "features": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
            "targets": torch.tensor([0.8, 0.2, 0.1], dtype=torch.float32),
        },
        {
            "features": torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32),
            "targets": torch.tensor([0.1, 0.7, 0.0], dtype=torch.float32),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate CORE frozen uncertainty pathway")
    parser.add_argument("--config", default="configs/paper_default.yaml")
    parser.add_argument("--calibration_file", default=None)
    parser.add_argument("--output_dir", default="outputs/uncertainty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    if args.dry_run:
        rows = dry_run_rows()
    else:
        if not args.calibration_file:
            parser.error("--calibration_file is required unless --dry_run is set")
        rows = load_calibration_jsonl(args.calibration_file)
    head, log = train_uncertainty_head(rows, config, seed=args.seed)
    log["dry_run"] = bool(args.dry_run)
    log["calibration_examples"] = len(rows)
    log["frozen_for_ppo"] = all(not param.requires_grad for param in head.parameters())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = state_dict_fingerprint(head.state_dict())
    log["uncertainty_head_fingerprint"] = fingerprint
    torch.save(
        {
            "config": config,
            "state_dict": head.state_dict(),
            "frozen_for_ppo": all(not param.requires_grad for param in head.parameters()),
            "uncertainty_head_fingerprint": fingerprint,
        },
        output_dir / "uncertainty_head.pt",
    )
    (output_dir / "uncertainty_calibration_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(log, indent=2))
    return 0


def _target_tensor(payload: dict[str, Any]) -> torch.Tensor:
    if "targets" in payload:
        target = payload["targets"]
    else:
        target = [payload["relevance"], payload["ambiguity"], payload["conflict"]]
    tensor = torch.tensor(target, dtype=torch.float32)
    if tensor.numel() != 3:
        raise ValueError("uncertainty calibration target must contain relevance, ambiguity, and conflict")
    return tensor.reshape(3)


def state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _zero_grad(head: FrozenUncertaintyHead) -> None:
    for param in head.parameters():
        param.grad = None


def _sgd_step(head: FrozenUncertaintyHead, learning_rate: float) -> None:
    with torch.no_grad():
        for param in head.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-learning_rate)


if __name__ == "__main__":
    raise SystemExit(main())
