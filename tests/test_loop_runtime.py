"""End-to-end loop tests: the runtime-native chain on MockEval.

These tests prove the LOOP IS THE EVENT LOG — every phase emits real
activegraph events, and the report is built by scanning the log.
"""

from __future__ import annotations

import dataclasses

import pytest

from regimes.agent import transforms as T
from regimes.eval.types import EvalResult, Outcome
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


# ---------------------------------------------------------------------------
# Confirm gate: held-out validation must gate promotion
# ---------------------------------------------------------------------------


def _promotable_optimize_instances():
    """OPTIMIZE instances where stub_topk_boost (assembly-crowding author)
    flips one failure to correct → promotion-eligible on OPTIMIZE."""
    return [
        MockInstance("q_ok1", "multi-session", False, ("s1",), True,
                     scores={"s1#0": 1.0}, selected_turn_ids=("s1#0",)),
        MockInstance("q_ok2", "multi-session", False, ("s2",), True,
                     scores={"s2#0": 0.9}, selected_turn_ids=("s2#0",)),
        MockInstance(
            "q_ac", "multi-session", False, ("sG",), False,
            scores={"sG#0": 0.6, "sN#0": 0.95},
            ranked=("sN#0", "sG#0"),
            selected_turn_ids=("sN#0",), truncated=True,
            gold_score_threshold=0.7,
            candidate_turn_ids=("sG#0",),
        ),
    ]


class _RegressOnConfirmEval:
    """MockEval wrapper that flips marked instances from correct to
    incorrect when transforms are installed — simulates overfitting
    where OPTIMIZE improves but CONFIRM regresses."""

    def __init__(self, regress_qids):
        self._base = MockEval()
        self._regress_qids = frozenset(regress_qids)

    def run_on_split(self, instances, **kw):
        result = self._base.run_on_split(instances, **kw)
        if not T.get_pipeline():
            return result
        new_outcomes = []
        for o in result.outcomes:
            if o.question_id in self._regress_qids:
                new_outcomes.append(
                    dataclasses.replace(o, correct=False, judge_label="mock-0")
                )
            else:
                new_outcomes.append(o)
        return EvalResult(
            outcomes=new_outcomes,
            aggregate=result.aggregate,
            backend=result.backend,
            run_dir=result.run_dir,
            config=result.config,
        )


def test_confirm_gate_promotes_when_positive_on_both():
    """A transform that improves both OPTIMIZE and CONFIRM still promotes."""
    optimize = _promotable_optimize_instances()
    confirm = [
        MockInstance("qc_ok1", "multi-session", False, ("sc1",), True,
                     scores={"sc1#0": 1.0}, selected_turn_ids=("sc1#0",)),
        MockInstance("qc_ok2", "multi-session", False, ("sc2",), True,
                     scores={"sc2#0": 0.9}, selected_turn_ids=("sc2#0",)),
        MockInstance(
            "qc_flip", "multi-session", False, ("scG",), False,
            scores={"scG#0": 0.6, "scN#0": 0.95},
            ranked=("scN#0", "scG#0"),
            selected_turn_ids=("scN#0",), truncated=True,
            gold_score_threshold=0.7,
            candidate_turn_ids=("scG#0",),
        ),
    ]
    rep = run_loop(eval_backend=MockEval(), instances=optimize,
                   confirm_instances=confirm)
    types = {e.type for e in rep.events}
    assert TRANSFORM_PROMOTED in types
    promo = rep.promotions[0]
    assert promo["confirm_delta"] is not None
    assert promo["confirm_delta"] > 0


def test_confirm_gate_discards_on_confirm_regression():
    """A transform that passes OPTIMIZE gates but regresses on CONFIRM
    is discarded with reason='confirm_regression'."""
    optimize = _promotable_optimize_instances()
    confirm = [
        MockInstance("qc_ok1", "multi-session", False, ("sc1",), True,
                     scores={"sc1#0": 1.0}, selected_turn_ids=("sc1#0",)),
        MockInstance("qc_ok2", "multi-session", False, ("sc2",), True,
                     scores={"sc2#0": 0.9}, selected_turn_ids=("sc2#0",)),
    ]
    backend = _RegressOnConfirmEval(regress_qids={"qc_ok1", "qc_ok2"})
    rep = run_loop(eval_backend=backend, instances=optimize,
                   confirm_instances=confirm, max_consecutive_discards=1)
    types = {e.type for e in rep.events}
    assert TRANSFORM_DISCARDED in types
    assert TRANSFORM_PROMOTED not in types
    discard = rep.discards[0]
    assert "confirm_regression" in discard["reasons"]
    assert discard["confirm_delta"] < 0
    assert "confirm_threshold" in discard


def test_confirm_gate_threshold_is_configurable():
    """When confirm_threshold is set above zero, a zero-delta CONFIRM
    result is treated as below threshold and discarded."""
    from regimes.targets.longmemeval import LongMemEvalTarget
    from regimes.targets.longmemeval.action_space import LongMemEvalActionSpace
    from regimes.targets.longmemeval.taxonomy import LongMemEvalTaxonomy

    optimize = _promotable_optimize_instances()
    confirm = [
        MockInstance("qc_ok1", "multi-session", False, ("sc1",), True,
                     scores={"sc1#0": 1.0}, selected_turn_ids=("sc1#0",)),
        MockInstance("qc_ok2", "multi-session", False, ("sc2",), True,
                     scores={"sc2#0": 0.9}, selected_turn_ids=("sc2#0",)),
    ]
    backend = MockEval()
    aspace = LongMemEvalActionSpace(confirm_threshold=0.05)
    target = LongMemEvalTarget(
        eval_backend=backend, action_space=aspace,
        taxonomy=LongMemEvalTaxonomy(),
    )
    rep = run_loop(target=target, instances=optimize,
                   confirm_instances=confirm, max_consecutive_discards=1)
    types = {e.type for e in rep.events}
    assert TRANSFORM_DISCARDED in types
    assert TRANSFORM_PROMOTED not in types
    discard = rep.discards[0]
    assert "confirm_regression" in discard["reasons"]
    assert discard["confirm_delta"] == pytest.approx(0.0)
    assert discard["confirm_threshold"] == pytest.approx(0.05)


def test_confirm_gate_noop_without_confirm_instances():
    """When no confirm instances are supplied, the gate is a no-op
    and the transform promotes as before."""
    optimize = _promotable_optimize_instances()
    rep = run_loop(eval_backend=MockEval(), instances=optimize,
                   confirm_instances=None)
    types = {e.type for e in rep.events}
    assert TRANSFORM_PROMOTED in types
    promo = rep.promotions[0]
    assert promo["confirm_delta"] is None


# ---------------------------------------------------------------------------
# Held-out persistence: per-question CONFIRM outcomes + bidirectional
# attribution (the analyzable held-out data the report must carry)
# ---------------------------------------------------------------------------


def _confirm_with_one_flip():
    """A 3-question CONFIRM set; the third flips correct under the
    promoted transform (assembly-crowding boost), so held-out gain is
    observable per-question."""
    return [
        MockInstance("qc_ok1", "multi-session", False, ("sc1",), True,
                     scores={"sc1#0": 1.0}, selected_turn_ids=("sc1#0",)),
        MockInstance("qc_ok2", "knowledge-update", False, ("sc2",), True,
                     scores={"sc2#0": 0.92}, selected_turn_ids=("sc2#0",)),
        MockInstance(
            "qc_flip", "multi-session", False, ("scG",), False,
            scores={"scG#0": 0.6, "scN#0": 0.95},
            ranked=("scN#0", "scG#0"),
            selected_turn_ids=("scN#0",), truncated=True,
            gold_score_threshold=0.7,
            candidate_turn_ids=("scG#0",),
        ),
    ]


def test_promotion_persists_per_question_confirm_outcomes():
    """A promoted transform persists per-question outcomes for BOTH
    baseline-on-CONFIRM and transform-on-CONFIRM, in the baseline shape
    (qid/type/correct/regime/is_abstention) — so per-type held-out
    deltas, flips and abstention movement are reconstructable."""
    optimize = _promotable_optimize_instances()
    confirm = _confirm_with_one_flip()
    rep = run_loop(eval_backend=MockEval(), instances=optimize,
                   confirm_instances=confirm)
    promo = rep.promotions[0]

    base = promo["confirm_baseline_outcomes"]
    xform = promo["confirm_transform_outcomes"]
    assert len(base) == len(confirm)
    assert len(xform) == len(confirm)
    # Same per-question audit shape as baseline.outcomes.
    for o in base + xform:
        assert {"question_id", "question_type", "correct", "regime",
                "is_abstention"} <= set(o)

    by_base = {o["question_id"]: o for o in base}
    by_xform = {o["question_id"]: o for o in xform}
    # The held-out flip is visible per-question: wrong in baseline, right
    # under the transform — NOT just an aggregate confirm_delta scalar.
    assert by_base["qc_flip"]["correct"] is False
    assert by_xform["qc_flip"]["correct"] is True


def test_attribution_records_bidirectional_optimize_and_confirm():
    """The attribution event carries direction-explicit transition rows
    for BOTH OPTIMIZE and CONFIRM, so regressions (right→wrong) are
    representable, not only wrong→right flips."""
    optimize = _promotable_optimize_instances()
    confirm = _confirm_with_one_flip()
    rep = run_loop(eval_backend=MockEval(), instances=optimize,
                   confirm_instances=confirm)
    att = rep.attributions[0]

    # OPTIMIZE rows carry explicit direction.
    assert att["split"] == "optimize"
    opt_rows = att["transition_rows"]
    assert opt_rows, "expected at least one OPTIMIZE transition"
    for r in opt_rows:
        assert r["status"] in {"gained", "lost", "shifted"}
        assert r["status"] == "gained" if r["after_correct"] else True
    assert any(r["status"] == "gained" for r in opt_rows)

    # CONFIRM (held-out) attribution present with the same explicit shape.
    conf_rows = att["confirm_transition_rows"]
    assert any(r["question_id"] == "qc_flip" and r["status"] == "gained"
               for r in conf_rows)
    assert att["confirm_n_recovered"] == 1
    assert att["confirm_n_introduced"] == 0


def test_directed_rows_mark_regressions_as_lost():
    """Direct unit check: a right→wrong transition is labeled 'lost' and
    counted as an introduced regression."""
    from regimes.loop.attribute import attribute

    def _o(qid, correct):
        return Outcome(question_id=qid, question_type="multi-session",
                       is_abstention=False, answer_session_ids=("s",),
                       correct=correct, scores={}, ranked=(),
                       selected_turn_ids=())

    before = EvalResult(outcomes=[_o("q1", True), _o("q2", False)],
                        aggregate={}, backend="mock")
    after = EvalResult(outcomes=[_o("q1", False), _o("q2", True)],
                       aggregate={}, backend="mock")
    att = attribute(before, after)
    rows = {r["question_id"]: r for r in att.directed_rows()}
    assert rows["q1"]["status"] == "lost"
    assert rows["q1"]["before_correct"] and not rows["q1"]["after_correct"]
    assert rows["q2"]["status"] == "gained"
    assert att.n_introduced == 1
    assert att.n_recovered == 1
