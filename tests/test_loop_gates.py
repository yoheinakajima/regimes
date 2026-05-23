"""Static + sandbox + eval-diff + promotion gates.

Gates are pure functions of their inputs; tests pin each safety
property and each promotion eligibility rule.
"""

from __future__ import annotations

import pytest

from regimes.agent import transforms as T
from regimes.loop.gates import (
    compile_transform,
    eval_diff,
    promotion_decision,
    sandbox_gate,
    static_gate,
)
from regimes.loop.mock_eval import MockEval, MockInstance


# ===========================================================================
# Static gate
# ===========================================================================


VALID = (
    "def transform(scores, graph, question, question_date):\n"
    "    return {t: s * 1.1 for t, s in scores.items()}\n"
)


def test_static_passes_valid_transform():
    r = static_gate(VALID)
    assert r.passed, r.reasons


def test_static_rejects_syntax_error():
    r = static_gate("def transform(scores, graph, question, question_date)\n  return scores")
    assert not r.passed
    assert any("syntax-error" in x for x in r.reasons)


def test_static_rejects_wrong_signature():
    src = "def transform(scores):\n    return scores\n"
    r = static_gate(src)
    assert not r.passed
    assert any("signature mismatch" in x for x in r.reasons)


def test_static_rejects_non_whitelisted_import():
    src = "import os\n" + VALID
    r = static_gate(src)
    assert not r.passed
    assert any("import outside whitelist" in x for x in r.reasons)


def test_static_allows_math_import():
    src = "import math\n" + VALID
    r = static_gate(src)
    assert r.passed, r.reasons


def test_static_rejects_dangerous_builtins():
    src = (
        "def transform(scores, graph, question, question_date):\n"
        "    eval('1+1')\n"
        "    return scores\n"
    )
    r = static_gate(src)
    assert not r.passed
    assert any("banned" in x for x in r.reasons)


def test_static_rejects_dunder_attr_access():
    src = (
        "def transform(scores, graph, question, question_date):\n"
        "    x = scores.__class__\n"
        "    return scores\n"
    )
    r = static_gate(src)
    assert not r.passed
    assert any("banned attribute" in x for x in r.reasons)


def test_static_rejects_top_level_statements():
    # Static gate's safety story is "imports + one def" only; any other
    # top-level statement would execute at compile-time.
    src = "x = 1\n" + VALID
    r = static_gate(src)
    assert not r.passed


def test_compile_transform_returns_callable():
    fn = compile_transform(VALID)
    assert callable(fn)
    out = fn({"a": 1.0, "b": 2.0}, None, "q", "")
    assert out == {"a": pytest.approx(1.1), "b": pytest.approx(2.2)}


# ===========================================================================
# Sandbox gate
# ===========================================================================


def test_sandbox_passes_on_well_behaved_transform():
    fn = compile_transform(VALID)
    probes = [{"scores": {"a": 1.0, "b": 2.0}, "question": "q", "question_date": ""}]
    r = sandbox_gate(fn, probes=probes)
    assert r.passed
    assert r.n_probed == 1


def test_sandbox_rejects_introduced_keys():
    src = (
        "def transform(scores, graph, question, question_date):\n"
        "    out = dict(scores)\n"
        "    out['evil_new'] = 1.0\n"
        "    return out\n"
    )
    fn = compile_transform(src)
    r = sandbox_gate(fn, probes=[{"scores": {"a": 1.0}}])
    assert not r.passed
    assert any("introduced unknown turn_ids" in x for x in r.reasons)


def test_sandbox_rejects_raising_transform():
    src = (
        "def transform(scores, graph, question, question_date):\n"
        "    raise ValueError('boom')\n"
    )
    fn = compile_transform(src)
    r = sandbox_gate(fn, probes=[{"scores": {"a": 1.0}}])
    assert not r.passed
    assert any("raised" in x for x in r.reasons)


def test_sandbox_rejects_non_dict_return():
    src = (
        "def transform(scores, graph, question, question_date):\n"
        "    return list(scores.items())\n"
    )
    fn = compile_transform(src)
    r = sandbox_gate(fn, probes=[{"scores": {"a": 1.0}}])
    assert not r.passed
    assert any("non-dict" in x for x in r.reasons)


# ===========================================================================
# Eval-diff gate
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_pipeline():
    T.clear()
    yield
    T.clear()


def _flippable_instances():
    return [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}, selected_turn_ids=("s_ok#0",)),
        MockInstance(
            "q_bt", "multi-session", False, ("sG",), False,
            scores={"sG#0": 0.7, "sN#0": 0.6},
            ranked=("sG#0", "sN#0"),
            selected_turn_ids=("sN#0",), truncated=True,
            decisions=({"turn_id": "sG#0", "included": False, "reason": "budget"},),
            candidate_turn_ids=("sG#0", "sN#0"),
            gold_score_threshold=1.0,
        ),
    ]


def test_eval_diff_shrinks_target_regime_when_transform_flips_failure():
    insts = _flippable_instances()
    ev = MockEval()
    baseline = ev.run_on_split(insts)
    # Baseline must show q_bt as budget-truncation.
    assert baseline.overall_accuracy() == 0.5

    src = (
        "def transform(scores, graph, question, question_date):\n"
        "    return {t: s * 2.0 for t, s in scores.items()}\n"
    )
    fn = compile_transform(src)
    diff = eval_diff(
        fn=fn, fn_name="doubler", target_regime="budget-truncation",
        baseline=baseline, eval_backend=ev, instances=insts,
    )
    # gold's score 0.7 → 1.4 ≥ 1.0 threshold → q_bt flips correct
    assert diff.overall_after == 1.0
    assert diff.overall_delta == pytest.approx(0.5)
    assert diff.target_delta < 0   # budget-truncation regime shrank


def test_eval_diff_records_per_question_transitions():
    insts = _flippable_instances()
    ev = MockEval()
    baseline = ev.run_on_split(insts)
    src = (
        "def transform(scores, graph, question, question_date):\n"
        "    return {t: s * 2.0 for t, s in scores.items()}\n"
    )
    fn = compile_transform(src)
    diff = eval_diff(
        fn=fn, fn_name="doubler", target_regime="budget-truncation",
        baseline=baseline, eval_backend=ev, instances=insts,
    )
    qids_changed = {t[0] for t in diff.transitions}
    assert "q_bt" in qids_changed
    # q_bt went from budget-truncation to correct.
    bt_trans = next(t for t in diff.transitions if t[0] == "q_bt")
    assert bt_trans[1] == "budget-truncation"
    assert bt_trans[2] == "correct"


def test_eval_diff_reverts_transform_after_run():
    """After eval_diff returns, the agent's pipeline must NOT contain
    the candidate (it was only injected for the diff call)."""
    insts = _flippable_instances()
    ev = MockEval()
    baseline = ev.run_on_split(insts)
    src = "def transform(scores, graph, question, question_date):\n    return scores\n"
    fn = compile_transform(src)
    eval_diff(
        fn=fn, fn_name="passthrough", target_regime="budget-truncation",
        baseline=baseline, eval_backend=ev, instances=insts,
    )
    pipeline_names = [e.name for e in T.get_pipeline()]
    assert "passthrough" not in pipeline_names


# ===========================================================================
# Promotion decision
# ===========================================================================


def _diff(*, target_delta=-1, overall_delta=0.02, per_type=None):
    from regimes.loop.gates import EvalDiff
    return EvalDiff(
        overall_before=0.8,
        overall_after=0.8 + overall_delta,
        overall_delta=overall_delta,
        per_type_delta=per_type or {"multi-session": 0.0, "temporal-reasoning": 0.0},
        regime_before={"budget-truncation": 3},
        regime_after={"budget-truncation": max(0, 3 + target_delta)},
        target_regime="budget-truncation",
        target_delta=target_delta,
    )


def test_promotion_eligible_when_target_shrinks_no_regression():
    d = _diff(target_delta=-1, overall_delta=0.02)
    decision = promotion_decision(d)
    assert decision.eligible


def test_promotion_rejected_when_target_did_not_shrink():
    d = _diff(target_delta=0, overall_delta=0.02)
    decision = promotion_decision(d)
    assert not decision.eligible
    assert any("did not shrink" in r for r in decision.reasons)


def test_promotion_rejected_when_multi_session_regressed():
    d = _diff(target_delta=-1, overall_delta=0.02,
              per_type={"multi-session": -0.1, "temporal-reasoning": 0.05})
    decision = promotion_decision(d)
    assert not decision.eligible
    assert any("multi-session regressed" in r for r in decision.reasons)


def test_promotion_rejected_when_overall_regressed():
    d = _diff(target_delta=-1, overall_delta=-0.05)
    decision = promotion_decision(d)
    assert not decision.eligible
    assert any("overall regressed" in r for r in decision.reasons)
