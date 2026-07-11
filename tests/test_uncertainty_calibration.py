import json

import pytest
import torch

from train.calibrate_uncertainty import main as calibrate_main


def test_uncertainty_calibration_cli_saves_frozen_head(tmp_path):
    calibration_file = tmp_path / "calibration.jsonl"
    calibration_file.write_text(
        "\n".join(
            [
                json.dumps({"features": [0.1, 0.2, 0.3, 0.4], "targets": [0.8, 0.2, 0.1]}),
                json.dumps({"features": [0.4, 0.3, 0.2, 0.1], "targets": [0.1, 0.7, 0.0]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "uncertainty"

    rc = calibrate_main(
        [
            "--config",
            "configs/paper_default.yaml",
            "--calibration_file",
            str(calibration_file),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    log = json.loads((output_dir / "uncertainty_calibration_log.json").read_text(encoding="utf-8"))
    assert log["dry_run"] is False
    assert log["calibration_examples"] == 2
    assert log["frozen_for_ppo"] is True
    assert log["uncertainty_head_fingerprint"]
    checkpoint = torch.load(output_dir / "uncertainty_head.pt", map_location="cpu")
    assert checkpoint["uncertainty_head_fingerprint"] == log["uncertainty_head_fingerprint"]


def test_uncertainty_calibration_requires_file_without_dry_run(tmp_path):
    with pytest.raises(SystemExit) as exc:
        calibrate_main(["--config", "configs/paper_default.yaml", "--output_dir", str(tmp_path)])

    assert exc.value.code == 2
