#!/usr/bin/env python3
"""Run CORE delivery preflight checks from one command."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from openrlhf.core.schemas import normalize_dialogue_record, normalize_profile_record, validate_core_config


OLD_METHOD_TOKENS = ("cog" + "nis", "Cog" + "nis", "COG" + "NIS")
DEFAULT_EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _load_yaml(path: str):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install it with: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_dialogue_dataset(path: str) -> Dict[str, object]:
    errors: List[str] = []
    phase_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    num_dialogues = 0
    num_turns = 0

    for line_no, row in _iter_jsonl(path):
        try:
            record = normalize_dialogue_record(row)
            num_dialogues += 1
            num_turns += len(record.turns)
            for turn in record.turns:
                phase_counts[turn.phase] = phase_counts.get(turn.phase, 0) + 1
                if turn.gold_action:
                    action = turn.gold_action.normalized_action()
                    action_counts[action] = action_counts.get(action, 0) + 1
        except Exception as exc:
            errors.append(f"line {line_no}: {exc}")

    return {
        "ok": not errors,
        "num_dialogues": num_dialogues,
        "num_turns": num_turns,
        "phase_counts": dict(sorted(phase_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "errors": errors,
    }


def validate_profile_dataset(path: str) -> Dict[str, object]:
    errors: List[str] = []
    num_records = 0
    empty_records = 0
    for line_no, row in _iter_jsonl(path):
        try:
            text = normalize_profile_record(row).strip()
            num_records += 1
            if not text:
                empty_records += 1
        except Exception as exc:
            errors.append(f"line {line_no}: {exc}")
    return {
        "ok": not errors and empty_records == 0,
        "num_records": num_records,
        "empty_records": empty_records,
        "errors": errors,
    }


def scan_old_method_names(root: str) -> List[str]:
    matches: List[str] = []
    for path in _iter_files(Path(root)):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for token in OLD_METHOD_TOKENS:
            if token in text:
                matches.append(str(path))
                break
    return sorted(matches)


def run_preflight(args) -> Dict[str, object]:
    report: Dict[str, object] = {}

    config = _load_yaml(args.config)
    validate_core_config(config, check_paths=args.check_paths)
    report["config"] = {"ok": True, "path": args.config, "check_paths": args.check_paths}

    if args.dialogue_dataset:
        report["dialogue_dataset"] = validate_dialogue_dataset(args.dialogue_dataset)
    if args.profile_dataset:
        report["profile_dataset"] = validate_profile_dataset(args.profile_dataset)
    if args.scan_old_name:
        matches = scan_old_method_names(args.root)
        report["old_method_name_scan"] = {"ok": len(matches) == 0, "matches": matches}

    report["ok"] = _all_ok(report)
    return report


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError:
                    yield line_no, line


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in DEFAULT_EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def _all_ok(report: Dict[str, object]) -> bool:
    for key, value in report.items():
        if key == "ok":
            continue
        if isinstance(value, dict) and value.get("ok") is False:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CORE config/data/name preflight checks")
    parser.add_argument("--config", default="examples/core_config.yaml")
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--dialogue-dataset", default="examples/datasets/core_dialogue_schema_example.jsonl")
    parser.add_argument("--profile-dataset", default=None)
    parser.add_argument("--scan-old-name", action="store_true", default=True)
    parser.add_argument("--no-scan-old-name", action="store_false", dest="scan_old_name")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    report = run_preflight(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
