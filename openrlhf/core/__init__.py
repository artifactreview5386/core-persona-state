"""
CORE multi-turn conversational alignment module.

Neural CORE classes are imported lazily so schema/config utilities remain
lightweight.
"""

from .schemas import CoreConfigError, validate_core_config

_LAZY_EXPORTS = {
    "ACTION_SPACE": ("core.router", "ACTION_SPACE"),
    "DISCOURSE_ACTIONS": ("core.discourse", "DISCOURSE_ACTIONS"),
    "BeliefState": ("core.belief", "BeliefState"),
    "BeliefUpdater": ("core.belief", "BeliefUpdater"),
    "DiscoursePolicy": ("core.discourse", "DiscoursePolicy"),
    "EvidenceScorer": ("core.evidence", "EvidenceScorer"),
    "RevisionGate": ("core.revision", "RevisionGate"),
    "UpdateRouter": ("core.router", "UpdateRouter"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "CoreConfigError",
    "validate_core_config",
    "ACTION_SPACE",
    "DISCOURSE_ACTIONS",
    "BeliefState",
    "BeliefUpdater",
    "EvidenceScorer",
    "UpdateRouter",
    "DiscoursePolicy",
    "RevisionGate",
    *_LAZY_EXPORTS.keys(),
]



