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


# ---------------------------------------------------------------------------
# Chaotic Mock Author
# ---------------------------------------------------------------------------
#
# MockTypedAuthor only ever produces clean candidates — which is exactly
# why the loop's iteration/rotation/stop gaps survived testing: every
# attempt passed the static + sandbox gates, so the static_rejected and
# sandbox_rejected control-flow branches were never exercised.
#
# ChaoticMockAuthor reproduces the REAL LLM author's failure mix. Each
# draft can be scripted to emit one of four candidate KINDS:
#
#   "promote"        — a transform that, against a flippable fixture,
#                      shrinks the targeted regime → promotes (or, with a
#                      confirm-regressing backend, is confirm-discarded).
#   "discard"        — a valid identity transform: passes static+sandbox
#                      but changes nothing → fails the OPTIMIZE gates.
#   "static_reject"  — syntactically invalid code (unclosed brace), the
#                      exact failure from the last real run → fails static.
#   "sandbox_reject" — valid syntax, passes static, but raises at runtime
#                      → fails the sandbox gate.
#
# Kinds are scripted per regime; each regime keeps its own cursor and the
# LAST kind repeats once a regime's script runs out (so an "all garbage"
# regime keeps producing garbage until it exhausts the consecutive-failure
# ceiling and rotates).

# The reconciliation marker the mock reader treats as "now answers
# correctly" (see test_reader_prompt_seam.RECONCILE_MARKER and the mock
# reader's reads_correct). A reader-prompt "promote" candidate injects it.
RECONCILE_MARKER = "prefer the most recent entry"


def _chaotic_source(transform_type: str, kind: str) -> tuple[str, str, str]:
    """Return (name, source, rationale) for a (type, kind) pair.

    The source is shaped for `transform_type` so it routes through the
    right action-space seam, and engineered to land on `kind`'s gate
    outcome regardless of the eval backend (except "promote", whose final
    promote-vs-confirm-discard verdict is decided by the fixture/backend)."""
    if kind == "static_reject":
        # Unclosed brace → SyntaxError → static gate rejects. Same shape
        # across all types (the parse fails before signature is checked).
        return (
            f"garbage_{transform_type.replace('-', '_')}",
            "def transform(scores, graph, question, question_date):\n"
            "    return {\n",   # <- unclosed brace: invalid syntax
            "Malformed code (unclosed brace) — reproduces the real author's "
            "static_rejected attempt.",
        )

    if transform_type == "assembly-transform":
        sig = "def transform(selected_turns, scores, question, question_date):"
        if kind == "sandbox_reject":
            return (
                "crash_assembly",
                f"{sig}\n    raise RuntimeError('boom in sandbox')\n",
                "Valid syntax but raises at runtime → sandbox_rejected.",
            )
        if kind == "promote":
            return (
                "reorder_by_score",
                f"{sig}\n"
                "    return sorted(selected_turns, key=lambda t: -scores.get(t, 0.0))\n",
                "Reorder selected turns by descending score.",
            )
        # discard: identity (no change → OPTIMIZE gates discard).
        return (
            "identity_assembly",
            f"{sig}\n    return list(selected_turns)\n",
            "Identity assembly transform → no improvement → discarded.",
        )

    if transform_type == "reader-prompt-transform":
        sig = "def transform(prompt_parts, question, question_date):"
        if kind == "sandbox_reject":
            return (
                "crash_reader",
                f"{sig}\n    raise RuntimeError('boom in sandbox')\n",
                "Valid syntax but raises at runtime → sandbox_rejected.",
            )
        if kind == "promote":
            return (
                "reconcile_instructions",
                f"{sig}\n"
                "    out = dict(prompt_parts)\n"
                "    out['instruction'] = out.get('instruction', '') + "
                f"' When evidence conflicts, {RECONCILE_MARKER}.'\n"
                "    return out\n",
                "Inject the reconciliation marker → flips assemble-internal "
                "failures correct → promotes.",
            )
        # discard: identity (same keys, no marker added → no flip).
        return (
            "identity_reader",
            f"{sig}\n    return dict(prompt_parts)\n",
            "Identity reader-prompt transform → no marker → discarded.",
        )

    # score-transform (default)
    sig = "def transform(scores, graph, question, question_date):"
    if kind == "sandbox_reject":
        return (
            "crash_score",
            f"{sig}\n    raise RuntimeError('boom in sandbox')\n",
            "Valid syntax but raises at runtime → sandbox_rejected.",
        )
    if kind == "promote":
        return (
            "topk_boost",
            f"{sig}\n"
            "    if not scores:\n"
            "        return scores\n"
            "    items = sorted(scores.items(), key=lambda kv: -kv[1])\n"
            "    out = {}\n"
            "    for i, (tid, s) in enumerate(items):\n"
            "        out[tid] = s * 1.5 if i < 5 else s\n"
            "    return out\n",
            "Boost top-5 scores by 50% to pull gold past the budget wall.",
        )
    # discard: identity (no change → OPTIMIZE gates discard).
    return (
        "identity_score",
        f"{sig}\n    return dict(scores)\n",
        "Identity score transform → no improvement → discarded.",
    )


@dataclass
class ChaoticMockAuthor:
    """Mock author that emits the real author's failure mix on a script.

    `by_regime` maps a regime name → an ordered list of candidate kinds
    ("promote" | "discard" | "static_reject" | "sandbox_reject"), consumed
    one per draft for that regime. Each regime has an independent cursor;
    when its list is exhausted the LAST kind repeats. `default_kind` is
    used for any regime not present in `by_regime`."""

    name: str = "chaotic-mock"
    by_regime: dict[str, list[str]] = None
    default_kind: str = "discard"
    _cursors: dict[str, int] = None

    def __post_init__(self):
        if self.by_regime is None:
            self.by_regime = {}
        if self._cursors is None:
            self._cursors = {}

    def _next_kind(self, regime: str) -> str:
        script = self.by_regime.get(regime)
        if not script:
            return self.default_kind
        i = self._cursors.get(regime, 0)
        kind = script[i] if i < len(script) else script[-1]
        self._cursors[regime] = i + 1
        return kind

    def draft_typed(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Outcome],  # noqa: ARG002
        transform_type: str,
    ) -> DraftedTransform:
        kind = self._next_kind(dominant_regime)
        n, src, rat = _chaotic_source(transform_type, kind)
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
        types = REGIME_TO_TYPES.get(dominant_regime, ("score-transform",))
        t = types[0] if types else "score-transform"
        return self.draft_typed(
            dominant_regime=dominant_regime,
            failures=failures,
            transform_type=t,
        )
