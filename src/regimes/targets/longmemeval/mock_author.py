"""Deterministic mock author that drafts all three transform types.

Used by tests and mock-mode runs to exercise the full three-type
machinery without an LLM. Each transform type has a small library of
pre-written source strings that pass the static gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from regimes.eval.types import Outcome
from regimes.loop.hypothesize import DraftedTransform
from regimes.targets.longmemeval.transform_types import REGIME_TO_TYPES


# ---------------------------------------------------------------------------
# Per-type stub libraries
# ---------------------------------------------------------------------------

_SCORE_LIBRARY: dict[str, tuple[str, str, str]] = {
    "assembly-crowding": (
        "stub_topk_boost",
        (
            "def transform(scores, graph, question, question_date):\n"
            "    if not scores:\n"
            "        return scores\n"
            "    items = sorted(scores.items(), key=lambda kv: -kv[1])\n"
            "    out = {}\n"
            "    for i, (tid, s) in enumerate(items):\n"
            "        if i < 5:\n"
            "            out[tid] = s * 1.25\n"
            "        else:\n"
            "            out[tid] = s\n"
            "    return out\n"
        ),
        "Boost top-5 scores by 25% to pull them past competing filler.",
    ),
    "budget-truncation": (
        "stub_demote_low",
        (
            "def transform(scores, graph, question, question_date):\n"
            "    if not scores:\n"
            "        return scores\n"
            "    vals = sorted(scores.values())\n"
            "    if not vals:\n"
            "        return scores\n"
            "    cutoff = vals[len(vals) // 2]\n"
            "    out = {}\n"
            "    for tid, s in scores.items():\n"
            "        out[tid] = s if s >= cutoff else s * 0.5\n"
            "    return out\n"
        ),
        "Halve below-median scores so bottom-half filler doesn't eat budget.",
    ),
}

_ASSEMBLY_LIBRARY: dict[str, tuple[str, str, str]] = {
    "assembly-crowding": (
        "stub_relevance_reorder",
        (
            "def transform(selected_turns, scores, question, question_date):\n"
            "    if not selected_turns:\n"
            "        return selected_turns\n"
            "    return sorted(selected_turns, key=lambda t: -scores.get(t, 0.0))\n"
        ),
        "Reorder turns by descending score (relevance-first ordering).",
    ),
    "budget-truncation": (
        "stub_top_half_only",
        (
            "def transform(selected_turns, scores, question, question_date):\n"
            "    if not selected_turns:\n"
            "        return selected_turns\n"
            "    ranked = sorted(selected_turns, key=lambda t: -scores.get(t, 0.0))\n"
            "    keep = max(1, len(ranked) // 2)\n"
            "    return ranked[:keep]\n"
        ),
        "Keep only the top-half by score, freeing budget for gold turns.",
    ),
}

_READER_LIBRARY: dict[str, tuple[str, str, str]] = {
    "assemble-internal": (
        "stub_reconcile_instructions",
        (
            "def transform(prompt_parts, question, question_date):\n"
            "    out = dict(prompt_parts)\n"
            "    instr = out.get('instruction', '')\n"
            "    reconcile = (\n"
            "        ' When evidence conflicts, prefer the most recent entry.'\n"
            "        ' Cite the specific turn that supports your answer.'\n"
            "    )\n"
            "    out['instruction'] = instr + reconcile\n"
            "    return out\n"
        ),
        "Add reconciliation instructions for contradictory evidence.",
    ),
}


# ---------------------------------------------------------------------------
# Mock Author
# ---------------------------------------------------------------------------


@dataclass
class MockTypedAuthor:
    """Deterministic author that drafts all three transform types.

    Implements `draft_typed(dominant_regime, failures, transform_type)`
    for the widened LongMemEvalActionSpace's selective-drafting path,
    and also implements the legacy `draft(dominant_regime, failures)` for
    backward compatibility."""

    name: str = "mock-typed"

    def draft_typed(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Outcome],  # noqa: ARG002
        transform_type: str,
    ) -> DraftedTransform:
        """Draft a transform of the specified type for the regime."""
        library = self._library_for_type(transform_type)
        if dominant_regime in library:
            n, src, rat = library[dominant_regime]
        else:
            # Fallback: pick the first entry in the library
            n, src, rat = next(iter(library.values()))
        return DraftedTransform(
            name=n,
            source=src,
            target_regime=dominant_regime,
            author=self.name,
            rationale=rat,
        )

    def draft(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Outcome],
    ) -> DraftedTransform:
        """Legacy path: select type from regime, then draft."""
        types = REGIME_TO_TYPES.get(dominant_regime, ("score-transform",))
        t = types[0] if types else "score-transform"
        return self.draft_typed(
            dominant_regime=dominant_regime,
            failures=failures,
            transform_type=t,
        )

    def _library_for_type(self, transform_type: str) -> dict[str, tuple[str, str, str]]:
        if transform_type == "score-transform":
            return _SCORE_LIBRARY
        elif transform_type == "assembly-transform":
            return _ASSEMBLY_LIBRARY
        elif transform_type == "reader-prompt-transform":
            return _READER_LIBRARY
        return _SCORE_LIBRARY
