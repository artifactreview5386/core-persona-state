#!/usr/bin/env python3
"""Create a clean reviewer-facing CORE artifact archive."""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable, List


REVIEW_FILES = [
    "README.md",
    "ARTIFACT_EVALUATION.md",
    "CODE_PAPER_ALIGNMENT.md",
    "DATA_FORMAT.md",
    "requirements.txt",
    "requirements-core.txt",
    "pyproject.toml",
    ".gitattributes",
    ".gitignore",
    ".github/workflows/core-offline.yml",
    "core",
    "train",
    "configs",
    "scripts",
    "tests",
    "openrlhf/__init__.py",
    "openrlhf/cli/__init__.py",
    "openrlhf/cli/core_offline_smoke.py",
    "openrlhf/cli/core_preflight.py",
    "openrlhf/core/__init__.py",
    "openrlhf/core/schemas.py",
    "openrlhf/cli/package_core_artifact.py",
    "openrlhf/cli/validate_core_config.py",
    "openrlhf/cli/validate_core_dataset.py",
    "examples/core_config.yaml",
    "examples/datasets/core_dialogue_schema_example.jsonl",
]

EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_NAMES: set[str] = set()


def iter_review_files(root: Path, entries: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for entry in entries:
        path = root / entry
        if not path.exists():
            raise FileNotFoundError(f"Review artifact entry does not exist: {entry}")
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and _include_file(child):
                    files.append(child)
        elif _include_file(path):
            files.append(path)
    return sorted(set(files))


def create_archive(root: Path, output: Path) -> dict:
    files = iter_review_files(root, REVIEW_FILES)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(root).as_posix())

    return {
        "ok": True,
        "output": str(output),
        "num_files": len(files),
        "files": [str(path.relative_to(root).as_posix()) for path in files],
    }


def _include_file(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a clean CORE reviewer artifact zip")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--output", default="dist/core_review_artifact.zip", help="Output zip path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    report = create_archive(root, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
