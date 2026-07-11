from pathlib import Path
import subprocess
import sys
import zipfile

import yaml

from openrlhf.cli.package_core_artifact import REVIEW_FILES, create_archive


DEFAULT_ROOTS = ("core", "train", "openrlhf/core", "configs", "scripts")


def test_default_paths_do_not_contain_nonpaper_mechanisms():
    forbidden = _forbidden_terms()
    text_by_path = _default_text_files()

    hits = []
    for path, text in text_by_path.items():
        if path.endswith("verify_paper_faithful.sh"):
            continue
        for term in forbidden:
            if term in text:
                hits.append((path, term))

    assert hits == []


def test_paper_default_reward_weights_are_exact():
    config = yaml.safe_load(Path("configs/paper_default.yaml").read_text(encoding="utf-8"))

    assert config["reward"] == {
        "action": 0.35,
        "state": 0.30,
        "answer": 0.25,
        "clarification": 0.10,
        "kl_beta": 0.05,
    }


def test_reviewer_artifact_excludes_nonpaper_auxiliary_modules():
    review_entries = set(REVIEW_FILES)

    allowed_entries = {
        "openrlhf/__init__.py",
        "openrlhf/cli/__init__.py",
        "openrlhf/cli/core_offline_smoke.py",
        "openrlhf/cli/core_preflight.py",
        "openrlhf/cli/package_core_artifact.py",
        "openrlhf/cli/validate_core_config.py",
        "openrlhf/cli/validate_core_dataset.py",
        "openrlhf/core/__init__.py",
        "openrlhf/core/schemas.py",
    }

    assert {entry for entry in review_entries if entry.startswith("openrlhf/")} == allowed_entries


def test_reviewer_artifact_is_import_self_contained(tmp_path):
    archive = tmp_path / "core_review_artifact.zip"
    extract_dir = tmp_path / "artifact"
    create_archive(Path(".").resolve(), archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import openrlhf.core.schemas; import core.pipeline; import train.reward; print('ok')",
        ],
        cwd=extract_dir,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _default_text_files() -> dict[str, str]:
    files = [Path("README.md")]
    for root in DEFAULT_ROOTS:
        files.extend(Path(root).rglob("*"))
    return {
        str(path).replace("\\", "/"): path.read_text(encoding="utf-8")
        for path in files
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".sh", ".txt"}
    }


def _forbidden_terms() -> tuple[str, ...]:
    return (
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
        "weekday",
        "weekend",
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
