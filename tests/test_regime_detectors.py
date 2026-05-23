"""Detectors are pure over Outcome — these tests pin each detector's
boundary cases against synthetic Outcomes. No agent, no eval, no
runtime.
"""

from __future__ import annotations

import pytest

from regimes.eval.types import Outcome
from regimes.loop.regimes import (
    REGIMES,
    SIGNAL_GAP_K,
    TOP_K,
    classify,
    detect_assemble_internal,
    detect_assembly_crowding,
    detect_budget_truncation,
    detect_retrieval_signal_gap,
    detect_scoring_error,
    histogram,
    register_regime,
    reset_regimes,
)


def mk(qid="q", *, correct=False, gold=("s1",), **kw) -> Outcome:
    """Minimal Outcome builder."""
    defaults = dict(
        question_id=qid,
        question_type="multi-session",
        is_abstention=False,
        answer_session_ids=tuple(gold),
        correct=correct,
        truncated=False,
        scores={},
        ranked=(),
        selected_turn_ids=(),
        decisions=(),
        score_error="",
    )
    defaults.update(kw)
    return Outcome(**defaults)


# ---------------------------------------------------------------------------
# scoring-error
# ---------------------------------------------------------------------------


def test_scoring_error_raised_in_behavior():
    o = mk(score_error="agent.score_embedding:BadRequestError: input too long")
    assert detect_scoring_error(o)


def test_scoring_error_gold_absent_from_scores_dict():
    # Gold session exists but no turn from any gold session appears in scores.
    o = mk(scores={"other#0": 0.5, "other#1": 0.2})
    assert detect_scoring_error(o)


def test_scoring_error_does_not_fire_when_gold_in_scores():
    # Gold IS in scores (even with value 0); that's signal-gap territory.
    o = mk(scores={"s1#0": 0.0, "other#0": 0.5})
    assert not detect_scoring_error(o)


def test_scoring_error_no_gold_sessions_does_not_fire():
    # Abstention questions can lack answer_session_ids — not a scoring error.
    o = mk(gold=())
    assert not detect_scoring_error(o)


# ---------------------------------------------------------------------------
# assemble-internal
# ---------------------------------------------------------------------------


def test_assemble_internal_fires_when_well_ranked_gold_mostly_selected_but_wrong():
    """New (coverage-based) semantics: assemble-internal requires
    well-ranked gold turns to be MOSTLY selected (coverage >= floor).
    Just "any gold-session turn selected" no longer triggers — that
    was the catch-all bug that bucketed every failure here."""
    o = mk(
        ranked=("s1#0", "s1#1", "other#0"),
        selected_turn_ids=("s1#0", "s1#1"),
        scores={"s1#0": 0.9, "s1#1": 0.85, "other#0": 0.5},
        correct=False,
    )
    # 2 of 2 well-ranked gold turns selected → coverage 1.0 → assemble-internal.
    assert detect_assemble_internal(o)


def test_assemble_internal_does_not_fire_when_correct():
    o = mk(
        ranked=("s1#0",),
        selected_turn_ids=("s1#0",),
        scores={"s1#0": 0.9},
        correct=True,
    )
    assert not detect_assemble_internal(o)


def test_assemble_internal_does_not_fire_on_partial_coverage():
    """The headline bug fix: 1 of 6 well-ranked gold turns selected
    (coverage 0.17) must NOT classify as assemble-internal — it's
    assembly-crowding. Pre-fix this test would have failed."""
    # gold ranked at positions 4,5,6,8,12,16 (six turns from gold session
    # s1), only one of them in selected.
    ranked = (
        "other#0", "other#1", "other#2", "other#3",
        "s1#0", "s1#1", "s1#2",
        "other#4",
        "s1#3",
        "other#5", "other#6", "other#7",
        "s1#4",
        "other#8", "other#9", "other#10",
        "s1#5",
    )
    scores = {t: 1.0 - i / 100 for i, t in enumerate(ranked)}
    o = mk(
        ranked=ranked,
        selected_turn_ids=("s1#0",),  # 1 of 6 selected
        scores=scores,
        truncated=True,
        correct=False,
    )
    assert not detect_assemble_internal(o)


def test_assemble_internal_does_not_fire_when_gold_not_selected():
    o = mk(selected_turn_ids=("other#0",))
    assert not detect_assemble_internal(o)


# ---------------------------------------------------------------------------
# budget-truncation
# ---------------------------------------------------------------------------


def test_budget_truncation_fires_with_decision_record():
    o = mk(
        truncated=True,
        decisions=(
            {"turn_id": "s1#0", "included": False, "reason": "budget"},
        ),
    )
    assert detect_budget_truncation(o)


def test_budget_truncation_requires_truncated_flag():
    # Same decision record but truncated=False: not a budget-truncation regime.
    o = mk(
        truncated=False,
        decisions=({"turn_id": "s1#0", "included": False, "reason": "budget"},),
    )
    assert not detect_budget_truncation(o)


def test_budget_truncation_requires_gold_in_decisions():
    o = mk(
        truncated=True,
        decisions=(
            {"turn_id": "other#0", "included": False, "reason": "budget"},
        ),
    )
    assert not detect_budget_truncation(o)


# ---------------------------------------------------------------------------
# assembly-crowding
# ---------------------------------------------------------------------------


def test_assembly_crowding_fires_when_gold_topk_not_selected():
    o = mk(
        ranked=("other#0", "s1#0", "other#1"),
        selected_turn_ids=("other#0",),
        scores={"other#0": 0.9, "s1#0": 0.7, "other#1": 0.5},
    )
    assert detect_assembly_crowding(o)


def test_assembly_crowding_does_not_fire_when_gold_selected():
    o = mk(
        ranked=("s1#0", "other#0"),
        selected_turn_ids=("s1#0",),
        scores={"s1#0": 0.9, "other#0": 0.5},
    )
    assert not detect_assembly_crowding(o)


def test_assembly_crowding_does_not_fire_when_gold_outside_topk():
    # gold at rank > TOP_K
    ranked = tuple(f"other#{i}" for i in range(TOP_K + 2)) + ("s1#0",)
    o = mk(ranked=ranked, selected_turn_ids=(), scores={t: 0.1 for t in ranked})
    assert not detect_assembly_crowding(o)


# ---------------------------------------------------------------------------
# retrieval-signal-gap
# ---------------------------------------------------------------------------


def test_signal_gap_fires_when_gold_outside_top_signal_gap_k():
    # Gold ranked at position SIGNAL_GAP_K (zero-indexed → not in top-K).
    ranked = tuple(f"other#{i}" for i in range(SIGNAL_GAP_K)) + ("s1#0",)
    scores = {t: 0.1 for t in ranked}
    scores["s1#0"] = 0.01
    o = mk(ranked=ranked, scores=scores)
    assert detect_retrieval_signal_gap(o)


def test_signal_gap_does_not_fire_when_gold_inside_window():
    ranked = ("other#0", "s1#0", "other#1")
    scores = {t: 0.5 for t in ranked}
    o = mk(ranked=ranked, scores=scores)
    assert not detect_retrieval_signal_gap(o)


def test_signal_gap_does_not_fire_when_gold_absent_from_scores():
    # Then scoring-error owns it, not signal-gap.
    o = mk(scores={"other#0": 0.5}, ranked=("other#0",))
    assert not detect_retrieval_signal_gap(o)


# ---------------------------------------------------------------------------
# classify priority order
# ---------------------------------------------------------------------------


def test_classify_priority_scoring_error_beats_others():
    # Gold IS in scores (would be signal-gap) but score_error is set —
    # scoring-error wins.
    ranked = tuple(f"other#{i}" for i in range(SIGNAL_GAP_K)) + ("s1#0",)
    scores = {t: 0.1 for t in ranked}
    o = mk(
        scores=scores, ranked=ranked,
        score_error="agent.score_embedding:BadRequestError: too long",
    )
    assert classify(o).name == "scoring-error"


def test_classify_assemble_internal_and_signal_gap_are_mutually_exclusive():
    """Under the coverage-based detector these regimes can never both
    match — assemble-internal requires well-ranked gold (non-empty
    top-K intersection); signal-gap requires the opposite. They can't
    pre-empt each other by construction."""
    # Well-ranked, mostly selected → assemble-internal.
    o_internal = mk(
        ranked=("s1#0",), selected_turn_ids=("s1#0",),
        scores={"s1#0": 0.9}, correct=False,
    )
    assert classify(o_internal).name == "assemble-internal"
    # Gold ranked far down → signal-gap, NOT assemble-internal.
    ranked = tuple(f"other#{i}" for i in range(25)) + ("s1#0",)
    scores = {t: 0.5 for t in ranked}
    scores["s1#0"] = 0.01
    o_gap = mk(ranked=ranked, scores=scores, selected_turn_ids=("other#0",),
               correct=False)
    assert classify(o_gap).name == "retrieval-signal-gap"


def test_classify_priority_assembly_crowding_beats_assemble_internal():
    """The headline reordering: with 1 of 6 well-ranked gold turns
    selected, the failure must NOT classify as assemble-internal even
    though one gold turn is in selected_turn_ids. assembly-crowding
    wins."""
    ranked = (
        "other#0", "other#1", "other#2", "other#3",
        "s1#0", "s1#1", "s1#2",
        "other#4",
        "s1#3",
        "other#5", "other#6", "other#7",
        "s1#4",
        "other#8", "other#9", "other#10",
        "s1#5",
    )
    scores = {t: 1.0 - i / 100 for i, t in enumerate(ranked)}
    o = mk(ranked=ranked, selected_turn_ids=("s1#0",), scores=scores,
           truncated=True, correct=False)
    assert classify(o).name == "assembly-crowding"


def test_classify_priority_budget_truncation_beats_assembly_crowding():
    # Set up overlap: gold in top-K AND a budget-reason decision.
    o = mk(
        truncated=True,
        ranked=("other#0", "s1#0"),
        selected_turn_ids=("other#0",),
        scores={"other#0": 0.9, "s1#0": 0.7},
        decisions=({"turn_id": "s1#0", "included": False, "reason": "budget"},),
    )
    assert classify(o).name == "budget-truncation"


def test_classify_unclassified_is_safety_floor():
    # An outcome that matches NO regime: incorrect, no gold session, no error.
    o = mk(gold=(), correct=False)
    # No gold → scoring-error skips, assemble-internal needs gold_selected (false).
    # Assembly-crowding needs gold → skips. Signal-gap needs gold → skips.
    assert classify(o).name == "unclassified"


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------


def test_histogram_counts_failures_per_regime():
    rows = histogram([
        mk("q_ok", correct=True, scores={"s1#0": 0.9}, selected_turn_ids=("s1#0",)),
        mk("q_se", score_error="x"),
        mk("q_se2", score_error="y"),
        # q_ai: gold ranked well AND mostly selected, but answer wrong.
        # Under the new coverage detector this is the assemble-internal
        # case — the agent retrieved correctly but the answer is wrong.
        mk("q_ai", scores={"s1#0": 0.9}, ranked=("s1#0",),
           selected_turn_ids=("s1#0",), correct=False),
    ])
    counts = {r.regime: r.count for r in rows}
    assert counts["scoring-error"] == 2
    assert counts["assemble-internal"] == 1
    # correct outcomes are excluded from failure-only histogram
    assert sum(r.count for r in rows) == 3


def test_histogram_full_includes_correct_only_when_requested():
    rows = histogram(
        [mk(correct=True, selected_turn_ids=("s1#0",))],
        failures_only=False,
    )
    # When failures_only=False, correct outcomes still get classified —
    # gold_selected=True ⇒ assemble-internal detector (but correct=True so
    # not really a failure). Note: histogram does not distinguish, it
    # just classifies. The intent is "regime distribution overall".
    counts = {r.regime: r.count for r in rows}
    assert sum(counts.values()) == 1


# ---------------------------------------------------------------------------
# extensibility hook
# ---------------------------------------------------------------------------


def test_register_regime_appends_to_taxonomy_and_priority():
    reset_regimes()
    try:
        def detector(o: Outcome) -> bool:
            return o.question_id == "q_new"
        register_regime(
            "new-regime", detector,
            optimizable=False, seam_reachable=False,
            description="test",
            priority_after="assembly-crowding",
        )
        assert "new-regime" in REGIMES()
        # Give scores containing gold so scoring-error doesn't pre-empt.
        o = mk(qid="q_new", correct=False, scores={"s1#0": 0.5})
        # Detector fires; new-regime classified. We inserted after
        # assembly-crowding (before retrieval-signal-gap), so new-regime
        # wins as long as no higher-priority built-in matches.
        result = classify(o)
        assert result.name == "new-regime"
    finally:
        reset_regimes()


def test_register_regime_rejects_duplicates():
    with pytest.raises(ValueError):
        register_regime(
            "scoring-error", lambda o: False,
            optimizable=False, seam_reachable=False,
        )
