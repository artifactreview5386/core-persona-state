"""
Schemas and validation helpers for CORE.

This module is intentionally lightweight: it does not import neural model
libraries. Use it in CLIs, notebooks, and pre-flight checks before training
runtime dependencies are available.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class CoreConfigError(ValueError):
    """Raised when a CORE config is structurally invalid."""


@dataclass(frozen=True)
class CoreActionLabel:
    """Gold or predicted routing action for one turn."""

    action: str
    slot: Optional[str] = None
    value: Optional[str] = None
    evidence: Optional[str] = None
    reason: Optional[str] = None

    def normalized_action(self) -> str:
        return (self.action or "").upper()


@dataclass(frozen=True)
class CoreTurnRecord:
    """Portable turn-level schema for CORE trajectories."""

    user: str
    assistant: Optional[str] = None
    phase: str = "interaction"
    gold_action: Optional[CoreActionLabel] = None
    ambiguity: bool = False
    conflict: bool = False
    pressure: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreDialogueRecord:
    """Portable dialogue-level schema for CORE trajectories."""

    dialogue_id: str
    persona_id: Optional[str]
    profile: str
    personality: str = "Helpful and consistent"
    core_axioms: List[str] = field(default_factory=list)
    turns: List[CoreTurnRecord] = field(default_factory=list)
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)


REQUIRED_CONFIG_SECTIONS = (
    "experiment",
    "model",
    "evidence",
    "uncertainty",
    "uncertainty_calibration",
    "belief",
    "thresholds",
    "sft",
    "reward",
    "ppo",
)
REQUIRED_MODEL_KEYS = ("context_dim", "slot_dim", "value_dim", "belief_dim", "hidden_dim", "vocab_size")
REQUIRED_REWARD_WEIGHTS = ("action", "state", "answer", "clarification", "kl_beta")
REQUIRED_SFT_KEYS = ("learning_rate", "batch_size", "epochs", "lambda_evi", "lambda_upd", "lambda_dis", "lambda_resp")
REQUIRED_PPO_KEYS = (
    "freeze_uncertainty_head",
    "clip_range",
    "value_coef",
    "entropy_coef",
    "gae_lambda",
    "discount",
    "effective_batch_size",
    "epochs_per_batch",
    "max_grad_norm",
    "reference_policy",
)


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise CoreConfigError(f"Config section '{key}' must be a mapping.")
    return value


def _require_keys(section: Mapping[str, Any], keys: Iterable[str], section_name: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise CoreConfigError(f"Config section '{section_name}' is missing keys: {', '.join(missing)}")


def validate_core_config(config: Mapping[str, Any], *, check_paths: bool = False) -> None:
    """
    Validate the public CORE YAML contract.

    Args:
        config: Parsed YAML dictionary.
        check_paths: If true, verify optional data paths exist on the current machine.

    Raises:
        CoreConfigError: If the structure is invalid.
    """
    if not isinstance(config, Mapping):
        raise CoreConfigError("CORE config must be a mapping.")

    _require_keys(config, REQUIRED_CONFIG_SECTIONS, "root")

    model = _require_mapping(config, "model")
    evidence = _require_mapping(config, "evidence")
    uncertainty = _require_mapping(config, "uncertainty")
    uncertainty_calibration = _require_mapping(config, "uncertainty_calibration")
    belief = _require_mapping(config, "belief")
    thresholds = _require_mapping(config, "thresholds")
    sft = _require_mapping(config, "sft")
    reward = _require_mapping(config, "reward")
    ppo = _require_mapping(config, "ppo")

    _require_keys(model, REQUIRED_MODEL_KEYS, "model")
    _require_keys(evidence, ("delta_max", "robust_scale_min_values"), "evidence")
    _require_keys(uncertainty, ("hidden_dim",), "uncertainty")
    _require_keys(uncertainty_calibration, ("learning_rate", "batch_size", "epochs"), "uncertainty_calibration")
    _require_keys(belief, ("new_value_prior",), "belief")
    _require_keys(thresholds, ("relevance",), "thresholds")
    _require_keys(sft, REQUIRED_SFT_KEYS, "sft")
    _require_keys(reward, REQUIRED_REWARD_WEIGHTS, "reward")
    _require_keys(ppo, REQUIRED_PPO_KEYS, "ppo")

    for key in REQUIRED_MODEL_KEYS:
        if int(model[key]) <= 0:
            raise CoreConfigError(f"model.{key} must be positive.")
    if float(evidence["delta_max"]) <= 0:
        raise CoreConfigError("evidence.delta_max must be positive.")
    if int(evidence["robust_scale_min_values"]) <= 0:
        raise CoreConfigError("evidence.robust_scale_min_values must be positive.")
    if int(uncertainty["hidden_dim"]) <= 0:
        raise CoreConfigError("uncertainty.hidden_dim must be positive.")
    if float(uncertainty_calibration["learning_rate"]) <= 0:
        raise CoreConfigError("uncertainty_calibration.learning_rate must be positive.")
    if int(uncertainty_calibration["batch_size"]) <= 0:
        raise CoreConfigError("uncertainty_calibration.batch_size must be positive.")
    if int(uncertainty_calibration["epochs"]) <= 0:
        raise CoreConfigError("uncertainty_calibration.epochs must be positive.")
    if float(belief["new_value_prior"]) <= 0:
        raise CoreConfigError("belief.new_value_prior must be positive.")
    if float(thresholds["relevance"]) < 0:
        raise CoreConfigError("thresholds.relevance must be non-negative.")
    if float(sft["learning_rate"]) <= 0:
        raise CoreConfigError("sft.learning_rate must be positive.")
    if int(sft["batch_size"]) <= 0:
        raise CoreConfigError("sft.batch_size must be positive.")
    if int(sft["epochs"]) <= 0:
        raise CoreConfigError("sft.epochs must be positive.")
    for key in ("lambda_evi", "lambda_upd", "lambda_dis", "lambda_resp"):
        if float(sft[key]) < 0:
            raise CoreConfigError(f"sft.{key} must be non-negative.")

    weight_sum = sum(float(reward[key]) for key in ("action", "state", "answer", "clarification"))
    if weight_sum <= 0:
        raise CoreConfigError("Reward weights must have a positive sum.")
    if float(reward["kl_beta"]) < 0:
        raise CoreConfigError("reward.kl_beta must be non-negative.")
    if not 0.0 < float(ppo["clip_range"]):
        raise CoreConfigError("ppo.clip_range must be positive.")
    if float(ppo["value_coef"]) < 0:
        raise CoreConfigError("ppo.value_coef must be non-negative.")
    if float(ppo["entropy_coef"]) < 0:
        raise CoreConfigError("ppo.entropy_coef must be non-negative.")
    if not 0.0 <= float(ppo["gae_lambda"]) <= 1.0:
        raise CoreConfigError("ppo.gae_lambda must be in [0, 1].")
    if not 0.0 < float(ppo["discount"]) <= 1.0:
        raise CoreConfigError("ppo.discount must be in (0, 1].")
    if int(ppo["effective_batch_size"]) <= 0:
        raise CoreConfigError("ppo.effective_batch_size must be positive.")
    if int(ppo["epochs_per_batch"]) <= 0:
        raise CoreConfigError("ppo.epochs_per_batch must be positive.")
    if float(ppo["max_grad_norm"]) <= 0:
        raise CoreConfigError("ppo.max_grad_norm must be positive.")
    if not str(ppo["reference_policy"]).strip():
        raise CoreConfigError("ppo.reference_policy must be non-empty.")

    if check_paths:
        optional_path_sections = [
            (name, config.get(name))
            for name in ("data_paths",)
            if isinstance(config.get(name), Mapping)
        ]
        for section_name, section in optional_path_sections:
            for key, value in section.items():
                if value and not Path(str(value)).exists():
                    raise CoreConfigError(f"{section_name}.{key} does not exist: {value}")


def normalize_profile_record(obj: Any) -> str:
    """Normalize profile/personality JSONL rows into plain text."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        for key in ("full_profile", "profile", "persona", "text", "content"):
            if obj.get(key):
                return str(obj[key])
        return str(dict(obj))
    if isinstance(obj, list):
        return "\n".join(str(item) for item in obj)
    return str(obj)


def parse_action_label(obj: Any) -> Optional[CoreActionLabel]:
    """Parse a flexible action label from dict/string records."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return CoreActionLabel(action=obj)
    if isinstance(obj, Mapping):
        return CoreActionLabel(
            action=str(obj.get("action", obj.get("type", "IGNORE"))),
            slot=obj.get("slot") or obj.get("key"),
            value=obj.get("value") or obj.get("val"),
            evidence=obj.get("evidence"),
            reason=obj.get("reason"),
        )
    return CoreActionLabel(action=str(obj))


def normalize_turn_record(obj: Mapping[str, Any]) -> CoreTurnRecord:
    """Normalize one CORE turn record."""
    user = obj.get("user") or obj.get("user_input") or obj.get("input") or obj.get("query")
    if not user:
        raise ValueError("Turn record must include user/user_input/input/query.")
    return CoreTurnRecord(
        user=str(user),
        assistant=obj.get("assistant") or obj.get("response"),
        phase=str(obj.get("phase", "interaction")),
        gold_action=parse_action_label(obj.get("gold_action") or obj.get("action_label")),
        ambiguity=bool(obj.get("ambiguity", obj.get("ambiguity_positive", False))),
        conflict=bool(obj.get("conflict", obj.get("conflict_positive", False))),
        pressure=bool(obj.get("pressure", obj.get("social_pressure", False))),
        metadata={k: v for k, v in obj.items() if k not in {"user", "user_input", "input", "query", "assistant", "response"}},
    )


def normalize_dialogue_record(obj: Mapping[str, Any]) -> CoreDialogueRecord:
    """Normalize one dialogue record into the CORE portable schema."""
    dialogue_id = str(obj.get("dialogue_id") or obj.get("id") or obj.get("session_id") or "unknown")
    turns_raw = obj.get("turns") or obj.get("dialogue") or obj.get("messages") or []
    turns = [normalize_turn_record(turn) for turn in turns_raw if isinstance(turn, Mapping)]
    core_axioms = obj.get("core_axioms") or obj.get("axioms") or []
    if isinstance(core_axioms, str):
        core_axioms = [core_axioms]
    return CoreDialogueRecord(
        dialogue_id=dialogue_id,
        persona_id=obj.get("persona_id"),
        profile=normalize_profile_record(obj.get("profile") or obj.get("persona") or obj),
        personality=str(obj.get("personality", "Helpful and consistent")),
        core_axioms=[str(item) for item in core_axioms],
        turns=turns,
        split=str(obj.get("split", "train")),
        metadata={k: v for k, v in obj.items() if k not in {"turns", "dialogue", "messages"}},
    )

