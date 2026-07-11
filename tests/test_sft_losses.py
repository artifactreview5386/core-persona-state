import json

import torch
import pytest

from train.sft import build_modules, compute_sft_losses, dry_run_batch, main as sft_main


def _config():
    return {
        "model": {
            "context_dim": 16,
            "slot_dim": 12,
            "value_dim": 10,
            "belief_dim": 8,
            "hidden_dim": 32,
            "vocab_size": 32,
        },
        "evidence": {"delta_max": 10.0, "robust_scale_min_values": 4},
        "belief": {"new_value_prior": 1e-6},
        "thresholds": {"relevance": 0.05},
        "sft": {
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 1,
            "lambda_evi": 1.0,
            "lambda_upd": 1.0,
            "lambda_dis": 1.0,
            "lambda_resp": 1.0,
        },
    }


def test_all_eq_15_19_losses_exist_and_backprop():
    config = _config()
    modules = build_modules(config)
    losses = compute_sft_losses(dry_run_batch(config), modules, config)

    for key in ("loss_bel", "loss_evi", "loss_upd", "loss_dis", "loss_resp", "loss_total"):
        assert key in losses
        assert torch.is_tensor(losses[key])

    losses["loss_total"].backward()
    assert any(param.grad is not None for param in modules.router.parameters())
    assert any(param.grad is not None for param in modules.discourse.parameters())


def test_masked_labels_keep_loss_term_present():
    config = _config()
    modules = build_modules(config)
    batch = dry_run_batch(config)
    batch["update_label_mask"].zero_()
    batch["discourse_label_mask"].zero_()
    losses = compute_sft_losses(batch, modules, config)

    assert losses["loss_upd"].item() == 0.0
    assert losses["loss_dis"].item() == 0.0
    assert "loss_total" in losses


def test_sft_cli_consumes_train_file(tmp_path):
    config = _config()
    batch = dry_run_batch(config)
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(_batch_to_json_rows(batch)) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    rc = sft_main(
        [
            "--config",
            "configs/paper_default.yaml",
            "--train_file",
            str(train_file),
            "--output_dir",
            str(output_dir),
            "--seed",
            "7",
        ]
    )

    assert rc == 0
    log = __import__("json").loads((output_dir / "sft_log.json").read_text(encoding="utf-8"))
    assert log["dry_run"] is False
    assert log["train_examples"] == 2
    assert (output_dir / "sft_modules.pt").exists()


def test_sft_cli_requires_train_file_without_dry_run(tmp_path):
    with pytest.raises(SystemExit) as exc:
        sft_main(["--config", "configs/paper_default.yaml", "--output_dir", str(tmp_path)])

    assert exc.value.code == 2


def test_sft_cli_requires_all_formal_supervision_fields(tmp_path):
    rows = [json.loads(line) for line in _batch_to_json_rows(dry_run_batch(_config()))]
    del rows[0]["gold_discourse"]
    train_file = tmp_path / "train_missing_label.jsonl"
    train_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        sft_main(
            [
                "--config",
                "configs/paper_default.yaml",
                "--train_file",
                str(train_file),
                "--output_dir",
                str(tmp_path / "out"),
            ]
        )


def _batch_to_json_rows(batch):
    rows = []
    batch_size = batch["context_repr"].size(0)
    batch_level = {
        "context_repr",
        "history_repr",
        "response_repr",
        "previous_belief_summary",
        "previous_belief",
        "value_mask",
        "gold_belief_value",
        "gold_evidence",
        "gold_update_action",
        "gold_discourse",
        "response_labels",
        "belief_label_mask",
        "evidence_label_mask",
        "update_label_mask",
        "discourse_label_mask",
        "response_label_mask",
    }
    for index in range(batch_size):
        row = {}
        for key, value in batch.items():
            tensor = value[index] if key in batch_level else value
            row[key] = tensor.tolist()
        rows.append(json.dumps(row))
    return rows
