#!/usr/bin/env python3
"""Paper-faithful supervised warm-up losses for CORE Eq.15-19."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from core.belief import BeliefUpdater
from core.discourse import DiscoursePolicy
from core.evidence import EvidenceScorer
from core.pipeline import run_core_step
from core.revision import RevisionGate
from core.router import UpdateRouter
from openrlhf.core.schemas import validate_core_config


MASK_KEYS = {"value_mask", "belief_label_mask", "evidence_label_mask", "update_label_mask", "discourse_label_mask", "response_label_mask"}
INDEX_KEYS = {"gold_belief_value", "gold_evidence", "gold_update_action", "gold_discourse", "response_labels"}
PADDED_SEQUENCE_KEYS = {"response_repr", "response_labels", "response_label_mask"}
FORMAL_SFT_REQUIRED_KEYS = (
    "context_repr",
    "history_repr",
    "slot_repr",
    "value_repr",
    "previous_belief_summary",
    "previous_belief",
    "value_mask",
    "gold_belief_value",
    "gold_evidence",
    "gold_update_action",
    "gold_discourse",
    "response_repr",
    "response_labels",
    "belief_label_mask",
    "evidence_label_mask",
    "update_label_mask",
    "discourse_label_mask",
    "response_label_mask",
)


@dataclass
class CoreSFTModules:
    evidence: EvidenceScorer
    router: UpdateRouter
    discourse: DiscoursePolicy
    revision_gate: RevisionGate
    belief_updater: BeliefUpdater
    response_head: nn.Linear

    def parameters(self):
        for module in (
            self.evidence,
            self.router,
            self.discourse,
            self.revision_gate,
            self.belief_updater,
            self.response_head,
        ):
            yield from module.parameters()


def build_modules(config: dict[str, Any]) -> CoreSFTModules:
    model = config["model"]
    evidence = EvidenceScorer(
        context_dim=int(model["context_dim"]),
        slot_dim=int(model["slot_dim"]),
        value_dim=int(model["value_dim"]),
        belief_dim=int(model["belief_dim"]),
        hidden_dim=int(model["hidden_dim"]),
    )
    router = UpdateRouter()
    discourse = DiscoursePolicy(hidden_dim=int(model["hidden_dim"]))
    revision_gate = RevisionGate()
    belief_updater = BeliefUpdater()
    response_head = nn.Linear(int(model["context_dim"]), int(model["vocab_size"]))
    return CoreSFTModules(evidence, router, discourse, revision_gate, belief_updater, response_head)


def compute_sft_losses(batch: dict[str, torch.Tensor], modules: CoreSFTModules, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Compute L_bel, L_evi, L_upd, L_dis, L_resp and total SFT loss."""
    step = run_core_step(
        batch,
        modules,
        relevance_threshold=float(config["thresholds"]["relevance"]),
        new_value_prior=float(config["belief"]["new_value_prior"]),
        evidence_delta_max=float(config["evidence"]["delta_max"]),
        evidence_robust_scale_min_values=int(config["evidence"]["robust_scale_min_values"]),
        teacher_force_actions=True,
    )
    response_repr = batch.get("response_repr", batch["context_repr"].unsqueeze(1))
    response_logits = modules.response_head(response_repr)

    loss_bel = _masked_nll_from_distribution(step.belief, batch.get("gold_belief_value"), batch.get("belief_label_mask"))
    loss_evi = _masked_cross_entropy(step.evidence_logits, batch.get("gold_evidence"), batch.get("evidence_label_mask"))
    loss_upd = _masked_cross_entropy(step.router.logits, batch.get("gold_update_action"), batch.get("update_label_mask"))
    loss_dis = _masked_cross_entropy(step.discourse.logits, batch.get("gold_discourse"), batch.get("discourse_label_mask"))
    loss_resp = _masked_cross_entropy(response_logits, batch.get("response_labels"), batch.get("response_label_mask"))

    lambdas = config["sft"]
    loss_total = (
        loss_bel
        + float(lambdas["lambda_evi"]) * loss_evi
        + float(lambdas["lambda_upd"]) * loss_upd
        + float(lambdas["lambda_dis"]) * loss_dis
        + float(lambdas["lambda_resp"]) * loss_resp
    )
    return {
        "loss_bel": loss_bel,
        "loss_evi": loss_evi,
        "loss_upd": loss_upd,
        "loss_dis": loss_dis,
        "loss_resp": loss_resp,
        "loss_total": loss_total,
        "mask_bel": _mask_count(batch.get("belief_label_mask"), loss_bel),
        "mask_evi": _mask_count(batch.get("evidence_label_mask"), loss_evi),
        "mask_upd": _mask_count(batch.get("update_label_mask"), loss_upd),
        "mask_dis": _mask_count(batch.get("discourse_label_mask"), loss_dis),
        "mask_resp": _mask_count(batch.get("response_label_mask"), loss_resp),
    }


def dry_run_batch(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    model = config["model"]
    batch, slots, values = 2, 3, 4
    return {
        "context_repr": torch.randn(batch, int(model["context_dim"])),
        "history_repr": torch.randn(batch, int(model["context_dim"])),
        "response_repr": torch.randn(batch, 3, int(model["context_dim"])),
        "slot_repr": torch.randn(slots, int(model["slot_dim"])),
        "value_repr": torch.randn(slots, values, int(model["value_dim"])),
        "null_option_repr": torch.randn(slots, int(model["value_dim"])),
        "previous_belief_summary": torch.randn(batch, slots, int(model["belief_dim"])),
        "previous_belief": torch.softmax(torch.randn(batch, slots, values), dim=-1),
        "value_mask": torch.ones(batch, slots, values, dtype=torch.bool),
        "gold_belief_value": torch.randint(0, values, (batch, slots)),
        "gold_evidence": torch.randint(0, values + 1, (batch, slots)),
        "gold_update_action": torch.randint(0, 3, (batch, slots)),
        "gold_discourse": torch.randint(0, 2, (batch,)),
        "response_labels": torch.randint(0, int(model["vocab_size"]), (batch, 3)),
        "belief_label_mask": torch.ones(batch, slots, dtype=torch.bool),
        "evidence_label_mask": torch.ones(batch, slots, dtype=torch.bool),
        "update_label_mask": torch.ones(batch, slots, dtype=torch.bool),
        "discourse_label_mask": torch.ones(batch, dtype=torch.bool),
        "response_label_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


def load_tensorized_jsonl(path: str) -> list[dict[str, torch.Tensor]]:
    """Load tensorized CORE SFT examples.

    Each JSONL row is one example with keys matching `compute_sft_losses`
    without a batch dimension. Raw text encoding is intentionally outside this
    trainer; released experiments should provide the tensorized representations
    produced by the paper preprocessing pipeline.
    """
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"no SFT records found in {path}")
    validate_sft_rows(records)
    return [{key: _json_value_to_tensor(key, value) for key, value in row.items()} for row in records]


def validate_sft_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = [key for key in FORMAL_SFT_REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"SFT row {index} is missing required keys: {', '.join(missing)}")


def collate_tensorized_examples(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    keys = set(examples[0])
    for example in examples[1:]:
        if set(example) != keys:
            missing = keys.symmetric_difference(example)
            raise ValueError(f"SFT examples must share the same tensor keys; mismatch: {sorted(missing)}")

    batch: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        tensors = [example[key] for example in examples]
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) == 1:
            batch[key] = torch.stack(tensors, dim=0)
        elif key in PADDED_SEQUENCE_KEYS:
            batch[key] = _pad_sequence_tensors(key, tensors)
        else:
            raise ValueError(f"cannot collate variable shapes for key {key}: {sorted(shapes)}")
    return batch


def train_sft(
    train_records: list[dict[str, torch.Tensor]],
    modules: CoreSFTModules,
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, float]:
    sft_cfg = config["sft"]
    epochs = int(sft_cfg["epochs"])
    batch_size = int(sft_cfg["batch_size"])
    learning_rate = float(sft_cfg["learning_rate"])

    first_batch = collate_tensorized_examples(train_records[:batch_size])
    with torch.no_grad():
        compute_sft_losses(first_batch, modules, config)

    rng = random.Random(seed)
    totals: dict[str, float] = {}
    steps = 0

    for _epoch in range(epochs):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            examples = [train_records[index] for index in order[start : start + batch_size]]
            batch = collate_tensorized_examples(examples)
            losses = compute_sft_losses(batch, modules, config)
            _zero_grad(modules)
            losses["loss_total"].backward()
            _sgd_step(modules, learning_rate)
            steps += 1
            for key, value in losses.items():
                if key.startswith("loss_") or key.startswith("mask_"):
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

    if steps == 0:
        raise ValueError("SFT training produced zero optimizer steps")
    return {key: value / steps for key, value in totals.items()} | {"train_steps": float(steps)}


def summarize_sft_records(
    records: list[dict[str, torch.Tensor]],
    modules: CoreSFTModules,
    config: dict[str, Any],
) -> dict[str, float]:
    sft_cfg = config["sft"]
    batch_size = int(sft_cfg["batch_size"])
    totals: dict[str, float] = {}
    steps = 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = collate_tensorized_examples(records[start : start + batch_size])
            losses = compute_sft_losses(batch, modules, config)
            steps += 1
            for key, value in losses.items():
                if key.startswith("loss_"):
                    totals[f"dev_{key}"] = totals.get(f"dev_{key}", 0.0) + float(value.detach().cpu())
    return {key: value / max(steps, 1) for key, value in totals.items()}


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
    if target is None:
        return logits.sum() * 0.0
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1).to(device=logits.device)
    losses = F.cross_entropy(flat_logits, flat_target, reduction="none")
    if mask is None:
        return losses.mean()
    flat_mask = mask.reshape(-1).to(device=logits.device, dtype=losses.dtype)
    denom = flat_mask.sum()
    if denom.item() == 0:
        return logits.sum() * 0.0
    return (losses * flat_mask).sum() / denom


def _masked_nll_from_distribution(distribution: torch.Tensor, target: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
    if target is None:
        return distribution.sum() * 0.0
    log_probs = distribution.clamp_min(1e-12).log()
    gathered = -log_probs.gather(-1, target.unsqueeze(-1).to(device=distribution.device)).squeeze(-1)
    if mask is None:
        return gathered.mean()
    mask = mask.to(device=distribution.device, dtype=gathered.dtype)
    denom = mask.sum()
    if denom.item() == 0:
        return distribution.sum() * 0.0
    return (gathered * mask).sum() / denom


def _mask_count(mask: torch.Tensor | None, loss: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones((), device=loss.device)
    return mask.to(device=loss.device, dtype=loss.dtype).sum()


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_core_config(config, check_paths=False)
    return config


def _json_value_to_tensor(key: str, value: Any) -> torch.Tensor:
    if key in MASK_KEYS:
        return torch.tensor(value, dtype=torch.bool)
    if key in INDEX_KEYS:
        return torch.tensor(value, dtype=torch.long)
    return torch.tensor(value, dtype=torch.float32)


def _pad_sequence_tensors(key: str, tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for tensor in tensors:
        pad_shape = (max_len - tensor.size(0), *tensor.shape[1:])
        if key in MASK_KEYS:
            pad = torch.zeros(pad_shape, dtype=torch.bool)
        else:
            pad = torch.zeros(pad_shape, dtype=tensor.dtype)
        padded.append(torch.cat([tensor, pad], dim=0))
    return torch.stack(padded, dim=0)


def _module_state_dict(modules: CoreSFTModules) -> dict[str, Any]:
    return {
        "evidence": modules.evidence.state_dict(),
        "router": modules.router.state_dict(),
        "discourse": modules.discourse.state_dict(),
        "revision_gate": modules.revision_gate.state_dict(),
        "response_head": modules.response_head.state_dict(),
    }


def _zero_grad(modules: CoreSFTModules) -> None:
    for param in modules.parameters():
        param.grad = None


def _sgd_step(modules: CoreSFTModules, learning_rate: float) -> None:
    with torch.no_grad():
        for param in modules.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-learning_rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CORE Eq.15-19 supervised warm-up")
    parser.add_argument("--config", default="configs/paper_default.yaml")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--dev_file", default=None)
    parser.add_argument("--output_dir", default="outputs/sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", default="dry-run-model")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    config = _load_yaml(args.config)
    modules = build_modules(config)
    if args.dry_run:
        batch = dry_run_batch(config)
        losses = compute_sft_losses(batch, modules, config)
        losses["loss_total"].backward()
        log = {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if key.startswith("loss_") or key.startswith("mask_")
        }
    else:
        if not args.train_file:
            parser.error("--train_file is required unless --dry_run is set")
        train_records = load_tensorized_jsonl(args.train_file)
        log = train_sft(train_records, modules, config, seed=args.seed)
        log["train_examples"] = len(train_records)
        if args.dev_file:
            dev_records = load_tensorized_jsonl(args.dev_file)
            log.update(summarize_sft_records(dev_records, modules, config))
            log["dev_examples"] = len(dev_records)
    log["model_name_or_path"] = args.model_name_or_path
    log["dry_run"] = bool(args.dry_run)
    log["train_file"] = args.train_file
    log["dev_file"] = args.dev_file
    log["label_policy"] = "Formal JSONL rows must provide Eq.15-19 labels and masks; zero masks keep a supervised term present without contributing loss."

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    if not log["dry_run"]:
        torch.save({"config": config, "state_dict": _module_state_dict(modules)}, output_dir / "sft_modules.pt")
    print(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
