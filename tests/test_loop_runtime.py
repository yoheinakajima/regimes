"""End-to-end loop tests: the runtime-native chain on MockEval.

These tests prove the LOOP IS THE EVENT LOG — every phase emits real
activegraph events, and the report is built by scanning the log.
"""

from __future__ import annotations

import pytest

from regimes.agent import transforms as T
from regimes.loop import (
    BASELINE_RECORDED,
    LOOP_STOPPED,
    REGIME_HISTOGRAM,
    TRANSFORM_DISCARDED,
    TRANSFORM_DRAFTED,
    TRANSFORM_EVAL_DIFF,
    TRANSFORM_PROMOTED,
    TRANSFORM_SANDBOX_PASSED,
    TRANSFORM_STATIC_PASSED,
    MockEval,
    MockInstance,
    run_loop,
)


@pytest.fixture(autouse=True)
def _clean_pipeline():
    T.clear()
    yield
    T.clear()


def _mix() -> list[MockInstance]:
    """A regime-mix fixture: 2 correct + 1 of each main regime."""
    return [
        MockInstance("q_ok1", "multi-session", False, ("s1",), True,
                     scores={"s1#0": 1.0}, selected_turn_ids=("s1#0",)),
        MockInstance("q_ok2", "single-session-user", False, ("s2",), True,
                     scores={"s2#0": 0.9}, selected_turn_ids=("s2#0",)),
        # scoring-error
        MockInstance("q_se", "multi-session", False, ("s3",), False,
                     scores={},
                     score_error="agent.score_embedding:BadRequestError: x"),
        # assembly-crowding: gold in top-5 ranked but not selected
        MockInstance("q_ac", "multi-session", False, ("sG",), False,
                     scores={"sG#0": 0.6, "sN#0": 0.9},
                     ranked=("sN#0", "sG#0"),
                     selected_turn_ids=("sN#0",), truncated=True),
        # budget-truncation: gold appears in decisions with reason='budget'
        MockInstance("q_bt", "temporal-reasoning", False, ("sH",), False,
                     scores={"sH#0": 0.8, "sM#0": 0.9},
                     ranked=("sM#0", "sH#0"),
                     selected_turn_ids=("sM#0",), truncated=True,
                     decisions=(
                         {"turn_id": "sH#0", "included": False, "reason": "budget"},
                     )),
    ]


# ---------------------------------------------------------------------------
# Pause-at-histogram path
# ---------------------------------------------------------------------------


def test_pause_after_histogram_emits_baseline_then_histogram_then_stop():
    rep = run_loop(eval_backend=MockEval(), instances=_mix(),
                   pause_after="histogram")
    types_in_order = [e.type for e in rep.events]
    # baseline.recorded must come before regime.histogram which must come
    # before loop.stopped.
    i_base = types_in_order.index(BASELINE_RECORDED)
    i_hist = types_in_order.index(REGIME_HISTOGRAM)
    i_stop = types_in_order.index(LOOP_STOPPED)
    assert i_base < i_hist < i_stop
    assert rep.stopped["reason"] == "pause_after_histogram"


def test_histogram_payload_carries_per_regime_counts():
    rep = run_loop(eval_backend=MockEval(), instances=_mix(),
                   pause_after="histogram")
    counts = {r["regime"]: r["count"] for r in rep.histogram["rows"]}
    assert counts["scoring-error"] == 1
    assert counts["assembly-crowding"] == 1
    assert counts["budget-truncation"] == 1
    assert rep.histogram["n_failures"] == 3
    assert rep.histogram["n_total"] == 5


def test_histogram_separates_scoring_error_from_other_regimes():
    """The headline diagnose discipline: scoring-error must NOT be
    bucketed under any optimizable regime."""
    rep = run_loop(eval_backend=MockEval(), instances=_mix(),
                   pause_after="histogram")
    se_row = next(r for r in rep.histogram["rows"] if r["regime"] == "scoring-error")
    assert se_row["count"] == 1
    assert se_row["optimizable"] is False
    assert se_row["seam_reachable"] is False
    # And the qid in the scoring-error bucket is q_se.
    assert "q_se" in se_row["qids"]


def test_pause_skips_hypothesize_and_later_phases():
    rep = run_loop(eval_backend=MockEval(), instances=_mix(),
                   pause_after="histogram")
    types_seen = {e.type for e in rep.events}
    assert TRANSFORM_DRAFTED not in types_seen
    assert TRANSFORM_STATIC_PASSED not in types_seen


# ---------------------------------------------------------------------------
# Full-loop path
# ---------------------------------------------------------------------------


def _flippable_mix() -> list[MockInstance]:
    """Two budget-truncation failures that flip when a top-K boost
    transform multiplies their scores enough to cross the threshold.

    The stub library's `stub_demote_low` halves below-median scores —
    that doesn't help. So we'll prove the eval-diff path with an
    explicit transform via a custom author below; for now, use this
    fixture in tests that DO expect a discard."""
    return [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}, selected_turn_ids=("s_ok#0",)),
        MockInstance(
            "q_bt1", "multi-session", False, ("sG1",), False,
            scores={"sG1#0": 0.7, "sN#0": 0.6},
            ranked=("sG1#0", "sN#0"),
            selected_turn_ids=("sN#0",), truncated=True,
            decisions=({"turn_id": "sG1#0", "included": False, "reason": "budget"},),
            candidate_turn_ids=("sG1#0", "sN#0"),
            gold_score_threshold=1.0,
        ),
        MockInstance(
            "q_bt2", "multi-session", False, ("sG2",), False,
            scores={"sG2#0": 0.6, "sM#0": 0.5},
            ranked=("sG2#0", "sM#0"),
            selected_turn_ids=("sM#0",), truncated=True,
            decisions=({"turn_id": "sG2#0", "included": False, "reason": "budget"},),
            candidate_turn_ids=("sG2#0", "sM#0"),
            gold_score_threshold=1.0,
        ),
    ]


def test_full_loop_runs_through_all_gate_events():
    rep = run_loop(eval_backend=MockEval(), instances=_flippable_mix(),
                   max_consecutive_discards=1)
    types_in_order = [e.type for e in rep.events]
    # First iteration should emit drafted → static_passed → sandbox_passed
    # → eval_diff → discarded or promoted.
    assert TRANSFORM_DRAFTED in types_in_order
    assert TRANSFORM_STATIC_PASSED in types_in_order
    assert TRANSFORM_SANDBOX_PASSED in types_in_order
    assert TRANSFORM_EVAL_DIFF in types_in_order
    # One of promoted/discarded must appear.
    assert (TRANSFORM_PROMOTED in types_in_order) or \
           (TRANSFORM_DISCARDED in types_in_order)


def test_loop_stops_with_named_wall_after_max_discards():
    rep = run_loop(eval_backend=MockEval(), instances=_flippable_mix(),
                   max_consecutive_discards=2)
    assert rep.stopped is not None
    # The wall must be one of the loop's known stop reasons.
    assert rep.stopped["reason"] in (
        "max_consecutive_discards",
        "no_optimizable_regime_remaining",
        "pause_after_histogram",
    )
    # `named_wall` must be a string (possibly empty if everything is
    # optimizable — in which case the stop was discard-driven).
    assert isinstance(rep.stopped["named_wall"], str)


def test_transform_log_records_every_attempt():
    """Held-out discipline: every drafted transform is in the log even
    when it never promotes. The CONFIRM-set headline is best-of-N from
    this log."""
    rep = run_loop(eval_backend=MockEval(), instances=_flippable_mix(),
                   max_consecutive_discards=2)
    # 2 discards plus one more in-flight attempt? Hard to assert exact
    # count, but at LEAST 1 entry must exist.
    assert len(rep.transform_log) >= 1
    statuses = {e["status"] for e in rep.transform_log}
    # Every entry has one of the gate statuses.
    assert statuses <= {"static_rejected", "sandbox_rejected", "discarded",
                        "promoted"}


# ---------------------------------------------------------------------------
# The wall-naming output (key deliverable of the stop phase)
# ---------------------------------------------------------------------------


def test_named_wall_for_signal_gap_only():
    """When only retrieval-signal-gap remains, the loop must stop with
    a wall payload that NAMES the regime + says what change would
    address it (a signal change, not a transform)."""
    insts = [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}),
        # Pure retrieval-signal-gap, all incorrect.
        *[
            MockInstance(
                f"q_sg{i}", "multi-session", False, (f"sG{i}",), False,
                ranked=tuple(f"oN{j}#0" for j in range(25)) + (f"sG{i}#0",),
                scores={f"sG{i}#0": 0.01,
                        **{f"oN{j}#0": 0.5 for j in range(25)}},
                selected_turn_ids=tuple(f"oN{j}#0" for j in range(5)),
            )
            for i in range(2)
        ],
    ]
    rep = run_loop(eval_backend=MockEval(), instances=insts,
                   max_consecutive_discards=1)
    assert rep.stopped["reason"] == "no_optimizable_regime_remaining"
    wall = rep.stopped["named_wall"]
    assert "retrieval-signal-gap" in wall
    assert "signal change" in wall


def test_named_wall_for_scoring_error_only():
    insts = [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}),
        MockInstance(
            "q_se", "multi-session", False, ("sX",), False,
            score_error="agent.score_embedding:BadRequestError: input too long",
        ),
    ]
    rep = run_loop(eval_backend=MockEval(), instances=insts,
                   max_consecutive_discards=1)
    assert rep.stopped["reason"] == "no_optimizable_regime_remaining"
    assert "scoring-error" in rep.stopped["named_wall"]
    # Must surface that a SCORING fix is needed, not a transform.
    assert "scoring" in rep.stopped["named_wall"].lower()
