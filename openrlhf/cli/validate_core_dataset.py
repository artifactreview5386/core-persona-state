#!/usr/bin/env python3
"""Validate CORE JSONL data without loading training runtime dependencies."""

import argparse
import json
import sys
from collections import Counter
from typing import Dict, List

from openrlhf.core.schemas import CoreDialogueRecord, normalize_dialogue_record, normalize_profile_record


ALLOWED_ACTIONS = {"COMMIT", "DEFER", "IGNORE"}


def _validate_dialogue(record: CoreDialogueRecord, line_no: int) -> List[str]:
    errors = []
    if not record.dialogue_id or record.dialogue_id == "unknown":
        errors.append(f"line {line_no}: missing dialogue_id/id/session_id")
    if not record.profile:
        errors.append(f"line {line_no}: missing profile/persona")
    if not record.turns:
        errors.append(f"line {line_no}: dialogue has no turns")

    for turn_idx, turn in enumerate(record.turns):
        if not turn.user.strip():
            errors.append(f"line {line_no} turn {turn_idx}: empty user text")
        if turn.gold_action:
            action = turn.gold_action.normalized_action()
            if action not in ALLOWED_ACTIONS:
                errors.append(f"line {line_no} turn {turn_idx}: invalid action {action!r}")
            if action in {"COMMIT", "DEFER"} and not turn.gold_action.slot:
                errors.append(f"line {line_no} turn {turn_idx}: {action} requires a slot")
            if action == "COMMIT" and not turn.gold_action.value:
                errors.append(f"line {line_no} turn {turn_idx}: COMMIT requires a value")
    return errors


def validate_dialogue_jsonl(path: str) -> Dict[str, object]:
    errors = []
    phase_counts = Counter()
    action_counts = Counter()
    num_dialogues = 0
    num_turns = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                record = normalize_dialogue_record(obj)
            except Exception as exc:
                errors.append(f"line {line_no}: cannot parse dialogue record: {exc}")
                continue

            num_dialogues += 1
            num_turns += len(record.turns)
            errors.extend(_validate_dialogue(record, line_no))
            for turn in record.turns:
                phase_counts[turn.phase] += 1
                if turn.gold_action:
                    action_counts[turn.gold_action.normalized_action()] += 1

    return {
        "ok": not errors,
        "num_dialogues": num_dialogues,
        "num_turns": num_turns,
        "phase_counts": dict(sorted(phase_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "errors": errors,
    }


def validate_profile_jsonl(path: str) -> Dict[str, object]:
    errors = []
    num_records = 0
    empty_records = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = line
            text = normalize_profile_record(obj).strip()
            num_records += 1
            if not text:
                empty_records += 1
                errors.append(f"line {line_no}: empty normalized profile text")

    return {
        "ok": not errors,
        "num_records": num_records,
        "empty_records": empty_records,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CORE dataset files")
    parser.add_argument("--path", required=True, help="Path to a JSONL dataset")
    parser.add_argument("--format", choices=["dialogue", "profile"], default="dialogue")
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    if args.format == "dialogue":
        summary = validate_dialogue_jsonl(args.path)
    else:
        summary = validate_profile_jsonl(args.path)

    visible = dict(summary)
    visible["errors"] = summary["errors"][: args.max_errors]
    print(json.dumps(visible, ensure_ascii=False, indent=2))

    if not summary["ok"]:
        print(f"Dataset validation failed with {len(summary['errors'])} error(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
