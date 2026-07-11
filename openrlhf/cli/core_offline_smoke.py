#!/usr/bin/env python3
"""Run dependency-light CORE smoke checks."""

import json

from openrlhf.core.schemas import normalize_dialogue_record, validate_core_config


def _load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for config validation. Install it with: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    config = _load_yaml("examples/core_config.yaml")
    validate_core_config(config, check_paths=True)

    with open("examples/datasets/core_dialogue_schema_example.jsonl", "r", encoding="utf-8") as f:
        record = normalize_dialogue_record(json.loads(f.readline()))
    assert record.dialogue_id == "core_0001"
    assert len(record.turns) == 3
    assert record.turns[1].gold_action.normalized_action() == "DEFER"

    print("CORE dependency-light smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
