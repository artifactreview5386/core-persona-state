#!/usr/bin/env python3
"""Validate a CORE YAML config without loading training runtime dependencies."""

import argparse
import sys

from openrlhf.core.schemas import CoreConfigError, validate_core_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CORE configuration")
    parser.add_argument("--config", required=True, help="Path to core_config.yaml")
    parser.add_argument("--check_paths", action="store_true", help="Also require optional data paths to exist")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print("PyYAML is required for YAML validation. Install it with: pip install pyyaml", file=sys.stderr)
        return 2

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        validate_core_config(config, check_paths=args.check_paths)
    except CoreConfigError as exc:
        print(f"Invalid CORE config: {exc}", file=sys.stderr)
        return 1

    print(f"CORE config is valid: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
