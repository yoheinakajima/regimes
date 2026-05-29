"""The prompt-transform pipeline for the SQL target.

Mirrors `regimes.agent.transforms` exactly in shape: an ordered list of
`(name, callable)` entries that the `sql_agent.prompt_pipeline` behavior
walks on every `columns.scored` event. Each entry's callable has
signature

    transform(prompt_parts: dict, question: str, schema_meta: dict) -> dict

and must return a new dict over the same keys as the input. The seam
guards this signature at the SQL ActionSpace's static-analysis gate
before promotion.

WHY ONE BEHAVIOR, NOT MANY: same as LME — registering N transforms as N
separate @behavior(on=["columns.scored"]) handlers would fork the
assembly chain. The pipeline keeps dispatch single-channel and records
the audit trail (`applied_transforms`) inside the one emitted event."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

PromptTransform = Callable[[dict, str, dict], dict]


@dataclass(frozen=True)
class PipelineEntry:
    name: str
    fn: PromptTransform


_LOCK = threading.Lock()
_PIPELINE: list[PipelineEntry] = []


def get_pipeline() -> list[PipelineEntry]:
    """Snapshot of the current pipeline."""
    with _LOCK:
        return list(_PIPELINE)


def promote(name: str, fn: PromptTransform) -> None:
    """Append a transform. The loop's promotion gate calls this only
    after all four lifecycle gates pass."""
    with _LOCK:
        _PIPELINE.append(PipelineEntry(name=name, fn=fn))


def revert(name: str) -> None:
    """Remove a transform by name."""
    with _LOCK:
        _PIPELINE[:] = [e for e in _PIPELINE if e.name != name]


def clear() -> None:
    """Drop every transform. Test isolation only."""
    with _LOCK:
        _PIPELINE.clear()


def apply_pipeline(
    *,
    prompt_parts: dict[str, Any],
    question: str,
    schema_meta: dict[str, Any],
) -> tuple[dict, list[dict]]:
    """Walk the pipeline. Returns (result, errors).

    `result` is {"prompt_parts": dict, "names": list[str]}.
    `errors` is a list of {"name": str, "error": str} for transforms
    that raised at runtime. A raising transform's contribution is
    skipped; the previous prompt_parts carry forward.
    """
    cur: dict[str, Any] = dict(prompt_parts)
    names: list[str] = []
    errors: list[dict] = []
    for entry in get_pipeline():
        try:
            new_parts = entry.fn(cur, question, schema_meta)
            if not isinstance(new_parts, dict):
                raise TypeError(
                    f"transform {entry.name!r} returned "
                    f"{type(new_parts).__name__}, expected dict"
                )
            extra = set(new_parts) - set(cur)
            if extra:
                raise ValueError(
                    f"transform {entry.name!r} introduced unknown "
                    f"prompt_parts keys: {sorted(extra)[:3]}..."
                )
            # Re-key to the same shape; missing keys keep their previous
            # value (transforms may filter, not invent).
            cur = {k: new_parts.get(k, cur[k]) for k in cur}
            names.append(entry.name)
        except Exception as e:  # noqa: BLE001 — agent runtime path
            errors.append({"name": entry.name, "error": f"{type(e).__name__}: {e}"})
    return ({"prompt_parts": cur, "names": names}, errors)
