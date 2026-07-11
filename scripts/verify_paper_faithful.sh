#!/usr/bin/env bash
set -euo pipefail

python -m pytest tests/

python - <<'PY'
from pathlib import Path
import sys
import yaml

roots = [Path("core"), Path("train"), Path("openrlhf/core"), Path("configs"), Path("scripts")]
files = [
    Path("README.md"),
    Path("ARTIFACT_EVALUATION.md"),
    Path("CODE_PAPER_ALIGNMENT.md"),
    Path("DATA_FORMAT.md"),
]
for root in roots:
    files.extend(path for path in root.rglob("*") if path.is_file())

skip = {str(Path("scripts/verify_paper_faithful.sh")).replace("\\", "/")}
suffixes = {".py", ".yaml", ".yml", ".md", ".sh", ".txt"}

terms = (
    "".join(("RE", "VISE")),
    "".join(("RE", "TRACT")),
    "".join(("DE", "LETE")),
    "".join(("SC", "OPE")),
    "_".join(("NO", "MEMORY")),
    "".join(("OVER", "RIDE")),
    "".join(("COR", "RECT")),
    "_".join(("evidence", "buffer")),
    "_".join(("deferred", "buffer")),
    "_".join(("defer", "accumulation")),
    "_".join(("graduate", "to", "commit")),
    "_".join(("revision", "timing")),
    "_".join(("memory", "rights")),
    "_".join(("context", "dependent")),
    "".join(("week", "day")),
    "".join(("week", "end")),
    "_".join(("dont", "remember")),
    "_".join(("do", "not", "remember")),
    "_".join(("meta", "instruction")),
    "_".join(("conflict", "threshold")),
    "".join(("X", "ML")),
    "".join(("x", "ml")),
    "".join(("<", "profile", ">")),
    "".join(("<", "/", "profile", ">")),
    "".join(("<", "response", ">")),
    "".join(("<", "/", "response", ">")),
    "".join(("reg", "ex")),
    "re." + "search",
    "re." + "match",
)

hits = []
for path in files:
    normalized = str(path).replace("\\", "/")
    if normalized in skip or path.suffix not in suffixes:
        continue
    text = path.read_text(encoding="utf-8")
    for term in terms:
        if term in text:
            hits.append((normalized, term))

if hits:
    for path, term in hits:
        print(f"forbidden term found: {path}: {term}", file=sys.stderr)
    raise SystemExit(1)

config = yaml.safe_load(Path("configs/paper_default.yaml").read_text(encoding="utf-8"))
expected_reward = {
    "action": 0.35,
    "state": 0.30,
    "answer": 0.25,
    "clarification": 0.10,
    "kl_beta": 0.05,
}
if config.get("reward") != expected_reward:
    raise SystemExit(f"paper_default reward mismatch: {config.get('reward')}")

required = [
    Path("ARTIFACT_EVALUATION.md"),
    Path("CODE_PAPER_ALIGNMENT.md"),
    Path("DATA_FORMAT.md"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing reviewer files: {', '.join(missing)}")

print("paper-faithful verification passed")
PY
