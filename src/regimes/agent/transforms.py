"""The score-transform pipeline.

Promotion writes here; the transform-seam behavior reads here. The
"pipeline" is an ordered list of (name, callable) entries. The default
empty list = passthrough.

The agent runtime fires `behavior_transform` (one registered behavior)
on every `turns.scored` event; the behavior body walks this pipeline
left-to-right. Each entry's callable has signature

    transform(scores: dict[str, float], graph, question: str,
              question_date: str) -> dict[str, float]

and must return a new dict keyed by the same turn_ids. The seam guards
this signature at the static-analysis gate before promotion (gates live
under regimes/gates/, separate from the agent).

WHY ONE BEHAVIOR, NOT MANY: registering N transforms as N separate
@behavior(on=["turns.scored"]) handlers would make the runtime emit N
turns.transformed events per question, which forks the assembly chain.
The pipeline keeps the dispatch single-channel and records the audit
trail (`applied_transforms`) inside the one emitted event.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

# transform signature, named for readability
ScoreTransform = Callable[[dict[str, float], Any, str, str], dict[str, float]]


@dataclass(frozen=True)
class PipelineEntry:
    name: str
    fn: ScoreTransform


_LOCK = threading.Lock()
_PIPELINE: list[PipelineEntry] = []


def get_pipeline() -> list[PipelineEntry]:
    """Snapshot of the current pipeline."""
    with _LOCK:
        return list(_PIPELINE)


def promote(name: str, fn: ScoreTransform) -> None:
    """Append a transform. The loop's promotion gate calls this only after
    all four lifecycle gates pass."""
    with _LOCK:
        _PIPELINE.append(PipelineEntry(name=name, fn=fn))


def revert(name: str) -> None:
    """Remove a transform by name. Used when a confirm-set check regresses
    or for test isolation."""
    with _LOCK:
        _PIPELINE[:] = [e for e in _PIPELINE if e.name != name]


def clear() -> None:
    """Drop every transform. Used at agent.retrieve() teardown for test
    isolation; the loop never calls this in production."""
    with _LOCK:
        _PIPELINE.clear()


def apply_pipeline(
    *,
    scores: dict[str, float],
    graph: Any,
    question_id: str,  # noqa: ARG001 — kept for parity with the seam contract
    question: str,
    question_date: str,
) -> tuple[dict, list[dict]]:
    """Walk the pipeline. Returns (result, errors).

    `result` is {"scores": dict, "names": list[str]}.
    `errors` is a list of {"name": str, "error": str} for transforms that
    raised at runtime. A raising transform's contribution is skipped; the
    previous scores carry forward. The static + sandbox gates exist
    precisely so live-pipeline errors are vanishingly rare, but we honor
    ActiveGraph's failure model: errors during runtime become event
    payload entries, not raises.
    """
    cur = dict(scores)
    names: list[str] = []
    errors: list[dict] = []
    for entry in get_pipeline():
        try:
            new_scores = entry.fn(cur, graph, question, question_date)
            if not isinstance(new_scores, dict):
                raise TypeError(
                    f"transform {entry.name!r} returned {type(new_scores).__name__}, expected dict"
                )
            # turn_id set must not grow (transforms may filter, not invent)
            extra = set(new_scores) - set(cur)
            if extra:
                raise ValueError(
                    f"transform {entry.name!r} introduced unknown turn_ids: {sorted(extra)[:3]}..."
                )
            cur = {tid: float(new_scores.get(tid, 0.0)) for tid in cur}
            names.append(entry.name)
        except Exception as e:  # noqa: BLE001 — agent runtime path; carried as event payload
            errors.append({"name": entry.name, "error": f"{type(e).__name__}: {e}"})
    return ({"scores": cur, "names": names}, errors)
