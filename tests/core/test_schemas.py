import subprocess
import sys

import pytest
import yaml

from openrlhf.core.schemas import (
    CoreConfigError,
    normalize_dialogue_record,
    normalize_profile_record,
    validate_core_config,
)


def _minimal_config():
    with open("configs/paper_default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_validate_core_config_accepts_minimal_contract():
    validate_core_config(_minimal_config())


def test_schema_import_stays_lightweight_without_torch_side_effect():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import openrlhf.core.schemas; print('torch' in sys.modules)",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "False"


def test_validate_core_config_rejects_missing_section():
    config = _minimal_config()
    del config["reward"]

    with pytest.raises(CoreConfigError):
        validate_core_config(config)


def test_validate_core_config_rejects_invalid_ppo_discount():
    config = _minimal_config()
    config["ppo"]["discount"] = 1.1

    with pytest.raises(CoreConfigError):
        validate_core_config(config)


def test_validate_core_config_rejects_nonpositive_sft_batch_size():
    config = _minimal_config()
    config["sft"]["batch_size"] = 0

    with pytest.raises(CoreConfigError):
        validate_core_config(config)


def test_validate_core_config_rejects_negative_relevance_threshold():
    config = _minimal_config()
    config["thresholds"]["relevance"] = -0.1

    with pytest.raises(CoreConfigError):
        validate_core_config(config)


def test_validate_core_config_rejects_zero_new_value_prior():
    config = _minimal_config()
    config["belief"]["new_value_prior"] = 0.0

    with pytest.raises(CoreConfigError):
        validate_core_config(config)


def test_normalize_profile_record_prefers_known_fields():
    assert normalize_profile_record({"full_profile": "likes tea", "other": "ignored"}) == "likes tea"
    assert normalize_profile_record(["a", "b"]) == "a\nb"


def test_normalize_dialogue_record_with_actions():
    record = normalize_dialogue_record(
        {
            "dialogue_id": "d1",
            "persona_id": "p1",
            "profile": "diet: vegetarian",
            "turns": [
                {
                    "phase": "anchoring",
                    "user": "I am vegetarian.",
                    "gold_action": {"action": "COMMIT", "slot": "diet", "value": "vegetarian"},
                },
                {
                    "phase": "fluctuation",
                    "user": "Maybe I am tired of it.",
                    "ambiguity": True,
                    "gold_action": {"action": "DEFER", "slot": "diet", "evidence": "temporary fatigue"},
                },
            ],
        }
    )

    assert record.dialogue_id == "d1"
    assert len(record.turns) == 2
    assert record.turns[0].gold_action.normalized_action() == "COMMIT"
    assert record.turns[1].ambiguity is True
