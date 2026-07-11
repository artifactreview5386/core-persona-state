from pathlib import Path
import json

import pytest
import torch

from train.ppo import compute_ppo_row, main as ppo_main, state_dict_fingerprint
from train.reward import compute_ppo_reward, format_reward_log, load_config, main as reward_main


def test_reward_weights_equal_paper_default_and_include_clarification():
    config = load_config("configs/paper_default.yaml")

    assert config["reward"]["action"] == 0.35
    assert config["reward"]["state"] == 0.30
    assert config["reward"]["answer"] == 0.25
    assert config["reward"]["clarification"] == 0.10
    assert config["reward"]["kl_beta"] == 0.05
    assert config["ppo"]["clip_range"] == 0.2
    assert config["ppo"]["value_coef"] == 0.5
    assert config["ppo"]["entropy_coef"] == 0.01
    assert config["ppo"]["gae_lambda"] == 0.95
    assert config["ppo"]["discount"] == 0.99

    reward, components = compute_ppo_reward(
        action=1.0,
        state=1.0,
        answer=1.0,
        clarification=1.0,
        kl=0.0,
        config=config,
    )

    assert reward == pytest.approx(1.0)
    assert set(components) == {"action", "state", "answer", "clarification", "kl"}
    log = format_reward_log(reward, components)
    assert set(log) == {
        "reward_action",
        "reward_state",
        "reward_answer",
        "reward_clarification",
        "reward_kl",
        "reward_total",
    }


def test_no_old_reward_triplet_in_training_code():
    train_files = list(Path("train").glob("*.py"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in train_files)
    old_action = ".".join(("0", "40"))
    old_state = ".".join(("0", "25"))
    old_answer = ".".join(("0", "35"))

    assert old_action not in text
    assert old_state not in text
    assert old_answer not in text


def test_ppo_row_uses_clipped_objective_and_config_reward():
    config = load_config("configs/paper_default.yaml")
    row = {
        "action": 1.0,
        "state": 1.0,
        "answer": 1.0,
        "clarification": 0.0,
        "kl": 0.0,
        "old_logprob": -2.0,
        "new_logprob": -1.0,
        "advantage": 1.0,
        "value": 0.0,
        "return": 1.0,
        "entropy": 0.0,
    }

    out = compute_ppo_row(row, config)

    assert out["reward_total"] == pytest.approx(0.9)
    assert out["ppo_clipped_ratio"] == pytest.approx(1.2)
    assert out["ppo_policy_loss"] == pytest.approx(-1.2)
    assert out["ppo_total_loss"] == pytest.approx(-0.7)


def test_ppo_and_reward_cli_require_input_files_without_dry_run(tmp_path):
    with pytest.raises(SystemExit) as ppo_exc:
        ppo_main(["--config", "configs/paper_default.yaml", "--output_dir", str(tmp_path / "ppo")])
    with pytest.raises(SystemExit) as reward_exc:
        reward_main(["--config", "configs/paper_default.yaml", "--output", str(tmp_path / "reward.jsonl")])

    assert ppo_exc.value.code == 2
    assert reward_exc.value.code == 2


def test_formal_reward_rows_require_all_reward_components(tmp_path):
    reward_input = tmp_path / "reward_rows.jsonl"
    reward_input.write_text(json.dumps({"action": 1.0, "state": 1.0}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        reward_main(
            [
                "--config",
                "configs/paper_default.yaml",
                "--input",
                str(reward_input),
                "--output",
                str(tmp_path / "reward.jsonl"),
            ]
        )


def test_ppo_cli_requires_and_accepts_frozen_uncertainty_head(tmp_path):
    head_state = {}
    fingerprint = state_dict_fingerprint(head_state)
    rollout_file = tmp_path / "rollouts.jsonl"
    rollout_file.write_text(
        json.dumps(
            {
                "action": 1.0,
                "state": 1.0,
                "answer": 1.0,
                "clarification": 0.0,
                "kl": 0.0,
                "old_logprob": -1.0,
                "new_logprob": -0.9,
                "advantage": 1.0,
                "value": 0.0,
                "return": 1.0,
                "entropy": 0.2,
                "uncertainty_head_fingerprint": fingerprint,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as missing_head_exc:
        ppo_main(
            [
                "--config",
                "configs/paper_default.yaml",
                "--rollout_file",
                str(rollout_file),
                "--output_dir",
                str(tmp_path / "missing_head"),
            ]
        )
    assert missing_head_exc.value.code == 2

    head_file = tmp_path / "uncertainty_head.pt"
    torch.save({"state_dict": head_state, "frozen_for_ppo": True, "uncertainty_head_fingerprint": fingerprint}, head_file)
    rc = ppo_main(
        [
            "--config",
            "configs/paper_default.yaml",
            "--rollout_file",
            str(rollout_file),
            "--uncertainty_head",
            str(head_file),
            "--output_dir",
            str(tmp_path / "ppo"),
        ]
    )

    assert rc == 0
    summary = json.loads((tmp_path / "ppo" / "ppo_summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is False
    assert summary["uncertainty_head"] == str(head_file)
    assert summary["uncertainty_head_fingerprint"] == fingerprint


def test_formal_ppo_rollouts_require_complete_objective_fields(tmp_path):
    rollout_file = tmp_path / "rollouts.jsonl"
    rollout_file.write_text(
        json.dumps(
            {
                "action": 1.0,
                "state": 1.0,
                "answer": 1.0,
                "clarification": 0.0,
                "kl": 0.0,
                "old_logprob": -1.0,
                "new_logprob": -0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    head_file = tmp_path / "uncertainty_head.pt"
    fingerprint = state_dict_fingerprint({})
    torch.save({"state_dict": {}, "frozen_for_ppo": True, "uncertainty_head_fingerprint": fingerprint}, head_file)

    with pytest.raises(ValueError, match="missing required keys"):
        ppo_main(
            [
                "--config",
                "configs/paper_default.yaml",
                "--rollout_file",
                str(rollout_file),
                "--uncertainty_head",
                str(head_file),
                "--output_dir",
                str(tmp_path / "ppo"),
            ]
        )


def test_formal_ppo_rollouts_must_match_frozen_uncertainty_head(tmp_path):
    rollout_file = tmp_path / "rollouts.jsonl"
    rollout_file.write_text(
        json.dumps(
            {
                "action": 1.0,
                "state": 1.0,
                "answer": 1.0,
                "clarification": 0.0,
                "kl": 0.0,
                "old_logprob": -1.0,
                "new_logprob": -0.9,
                "advantage": 1.0,
                "value": 0.0,
                "return": 1.0,
                "entropy": 0.2,
                "uncertainty_head_fingerprint": "wrong",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    head_file = tmp_path / "uncertainty_head.pt"
    fingerprint = state_dict_fingerprint({})
    torch.save({"state_dict": {}, "frozen_for_ppo": True, "uncertainty_head_fingerprint": fingerprint}, head_file)

    with pytest.raises(ValueError, match="does not match"):
        ppo_main(
            [
                "--config",
                "configs/paper_default.yaml",
                "--rollout_file",
                str(rollout_file),
                "--uncertainty_head",
                str(head_file),
                "--output_dir",
                str(tmp_path / "ppo"),
            ]
        )
