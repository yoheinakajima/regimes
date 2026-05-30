"""The reader-prompt-transform pipeline.

Mirrors `regimes.agent.transforms` (the score-transform pipeline) and
`regimes.targets.sql.prompt_transforms` (the SQL prompt-edit pipeline)
exactly in shape: an ordered list of `(name, callable)` entries.

The seam this pipeline feeds is the READER PROMPT — the fragments the
reader sees when it answers a question. A reader-prompt-transform has
signature

    transform(prompt_parts: dict, question: str, question_date: str) -> dict

and must return a new dict over the SAME keys as the input (it may edit
the text of a fragment, but not invent or drop fragments). The seam
guards this signature at the LongMemEval ActionSpace's static-analysis
gate before promotion.

WHY THIS MODULE EXISTS: the action space could install a
reader-prompt-transform, but nothing on the eval path ever READ it —
only the score-transform pipeline (`regimes.agent.transforms`) was
applied during `run_on_split`. So a drafted reader-prompt-transform
installed but never reached the reader, its eval-diff was always a
no-op, and it was always discarded. This pipeline is the seam both the
mock eval and the real reader path apply, so an installed
reader-prompt-transform actually changes the prompt the reader sees.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

# transform signature, named for readability
ReaderPromptTransform = Callable[[dict, str, str], dict]


@dataclass(frozen=True)
class PipelineEntry:
    name: str
    fn: ReaderPromptTransform


_LOCK = threading.Lock()
_PIPELINE: list[PipelineEntry] = []


def get_pipeline() -> list[PipelineEntry]:
    """Snapshot of the current pipeline."""
    with _LOCK:
        return list(_PIPELINE)


def promote(name: str, fn: ReaderPromptTransform) -> None:
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
    """Drop every transform. Test isolation only."""
    with _LOCK:
        _PIPELINE.clear()


def apply_pipeline(
    *,
    prompt_parts: dict[str, Any],
    question: str,
    question_date: str,
) -> tuple[dict, list[dict]]:
    """Walk the pipeline. Returns (result, errors).

    `result` is {"prompt_parts": dict, "names": list[str]}.
    `errors` is a list of {"name": str, "error": str} for transforms that
    raised at runtime. A raising transform's contribution is skipped; the
    previous prompt_parts carry forward. The static + sandbox gates exist
    precisely so live-pipeline errors are vanishingly rare, but we honor
    ActiveGraph's failure model: errors during runtime become event
    payload entries, not raises.
    """
    cur: dict[str, Any] = dict(prompt_parts)
    names: list[str] = []
    errors: list[dict] = []
    for entry in get_pipeline():
        try:
            new_parts = entry.fn(cur, question, question_date)
            if not isinstance(new_parts, dict):
                raise TypeError(
                    f"transform {entry.name!r} returned "
                    f"{type(new_parts).__name__}, expected dict"
                )
            # key set must not grow (transforms may edit, not invent)
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
