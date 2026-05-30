"""Per-transform-type configuration for the widened LongMemEval action space.

Three transform types, each with its own call shape, static-gate
whitelist, install/revert seam, and value validator:

  score-transform       — (scores, graph, question, question_date) -> dict
  assembly-transform    — (selected_turns, scores, question, question_date) -> list
  reader-prompt-transform — (prompt_parts, question, question_date) -> dict

Each type is represented by a TransformType descriptor that carries the
per-type gate configuration the generic action-space machinery needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

READER_PROMPT_MAX_ADDED_CHARS = 2000


# ---------------------------------------------------------------------------
# TransformType descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformType:
    """Per-type gate configuration."""

    name: str
    signature_params: tuple[str, ...]
    import_whitelist: frozenset[str]
    expected_fn: str

    # sandbox_gate hooks
    call_fn: Callable[[Callable, Mapping[str, Any]], Any]
    value_validator: Callable[[Any, Mapping[str, Any]], None]

    # probe key used to extract input-key set for the "no unknown keys"
    # invariant in sandbox_gate
    probe_input_key: str


# ---------------------------------------------------------------------------
# score-transform (EXISTING)
# ---------------------------------------------------------------------------


def _score_call_fn(fn: Callable, probe: Mapping[str, Any]) -> dict:
    input_scores = dict(probe.get("scores", {}))
    return fn(input_scores, None, probe.get("question", ""), probe.get("question_date", ""))


def _score_value_validator(out: Any, probe: Mapping[str, Any]) -> None:  # noqa: ARG001
    if not isinstance(out, dict):
        raise TypeError(f"expected dict, got {type(out).__name__}")
    {tid: float(v) for tid, v in out.items()}


SCORE_TRANSFORM = TransformType(
    name="score-transform",
    signature_params=("scores", "graph", "question", "question_date"),
    import_whitelist=frozenset({"math"}),
    expected_fn="transform",
    call_fn=_score_call_fn,
    value_validator=_score_value_validator,
    probe_input_key="scores",
)


# ---------------------------------------------------------------------------
# assembly-transform (NEW)
# ---------------------------------------------------------------------------


def _assembly_call_fn(fn: Callable, probe: Mapping[str, Any]) -> Any:
    selected_turns = list(probe.get("selected_turns", []))
    scores = dict(probe.get("scores", {}))
    return fn(selected_turns, scores, probe.get("question", ""), probe.get("question_date", ""))


def _assembly_value_validator(out: Any, probe: Mapping[str, Any]) -> None:
    if not isinstance(out, list):
        raise TypeError(f"expected list, got {type(out).__name__}")
    input_ids = set(probe.get("selected_turns", []))
    for tid in out:
        if not isinstance(tid, str):
            raise TypeError(f"list elements must be str, got {type(tid).__name__}")
        if tid not in input_ids:
            raise ValueError(f"fabricated turn_id not in input: {tid!r}")


ASSEMBLY_TRANSFORM = TransformType(
    name="assembly-transform",
    signature_params=("selected_turns", "scores", "question", "question_date"),
    import_whitelist=frozenset({"math", "string"}),
    expected_fn="transform",
    call_fn=_assembly_call_fn,
    value_validator=_assembly_value_validator,
    probe_input_key="selected_turns",
)


# ---------------------------------------------------------------------------
# reader-prompt-transform (NEW)
# ---------------------------------------------------------------------------


def _reader_call_fn(fn: Callable, probe: Mapping[str, Any]) -> Any:
    prompt_parts = dict(probe.get("prompt_parts", {}))
    return fn(prompt_parts, probe.get("question", ""), probe.get("question_date", ""))


def _reader_value_validator(out: Any, probe: Mapping[str, Any]) -> None:
    if not isinstance(out, dict):
        raise TypeError(f"expected dict, got {type(out).__name__}")
    input_keys = set(probe.get("prompt_parts", {}).keys())
    out_keys = set(out.keys())
    if out_keys != input_keys:
        extra = out_keys - input_keys
        missing = input_keys - out_keys
        parts = []
        if extra:
            parts.append(f"fabricated keys: {sorted(extra)}")
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        raise ValueError(f"must return same keys as input; {'; '.join(parts)}")
    # Length cap: reject if added content exceeds the threshold
    input_parts = probe.get("prompt_parts", {})
    total_added = 0
    for k in out:
        orig = str(input_parts.get(k, ""))
        new = str(out[k])
        if len(new) > len(orig):
            total_added += len(new) - len(orig)
    if total_added > READER_PROMPT_MAX_ADDED_CHARS:
        raise ValueError(
            f"added {total_added} chars exceeds cap of "
            f"{READER_PROMPT_MAX_ADDED_CHARS}"
        )


READER_PROMPT_TRANSFORM = TransformType(
    name="reader-prompt-transform",
    signature_params=("prompt_parts", "question", "question_date"),
    import_whitelist=frozenset({"math", "string"}),
    expected_fn="transform",
    call_fn=_reader_call_fn,
    value_validator=_reader_value_validator,
    probe_input_key="prompt_parts",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TRANSFORM_TYPES: dict[str, TransformType] = {
    t.name: t for t in [SCORE_TRANSFORM, ASSEMBLY_TRANSFORM, READER_PROMPT_TRANSFORM]
}

# Regime → eligible transform types
REGIME_TO_TYPES: dict[str, tuple[str, ...]] = {
    "budget-truncation": ("score-transform", "assembly-transform"),
    "assembly-crowding": ("score-transform", "assembly-transform"),
    "assemble-internal": ("reader-prompt-transform",),
    "retrieval-signal-gap": (),  # wall — no transform type reaches it
    "scoring-error": (),
    "unclassified": (),
}
