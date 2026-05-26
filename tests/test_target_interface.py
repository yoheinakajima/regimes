"""Phase 1 acceptance tests for the Target interface.

Two things to verify:

  1. LongMemEvalTarget structurally satisfies the Target protocol — its
     four sub-components implement the documented contracts.

  2. run_loop() works when handed a Target explicitly (not just the
     legacy `eval_backend=...` form), and produces a LoopReport with
     the same overall shape (baseline, histogram, stopped, etc.).
"""

from __future__ import annotations

from regimes.loop import MockEval, MockInstance, run_loop
from regimes.target import (
    ActionSpace,
    EvalBackend,
    RegimeTaxonomy,
    Target,
)
from regimes.targets.longmemeval import (
    LongMemEvalActionSpace,
    LongMemEvalTarget,
    LongMemEvalTaxonomy,
    build_target,
)


# ---------------------------------------------------------------------------
# Fixture: a small MockInstance mix shared between tests.
# ---------------------------------------------------------------------------


def _mini_split() -> list[MockInstance]:
    return [
        MockInstance("q_ok", "single-session-user", False, ("s1",), True,
                     scores={"s1#0": 1.0}, selected_turn_ids=("s1#0",)),
        MockInstance(
            "q_ac", "multi-session", False, ("sG",), False,
            scores={"sG#0": 0.6, "sN#0": 0.95},
            ranked=("sN#0", "sG#0"),
            selected_turn_ids=("sN#0",), truncated=True,
            gold_score_threshold=0.7,
            candidate_turn_ids=("sG#0",),
        ),
    ]


# ---------------------------------------------------------------------------
# 1) Protocol conformance
# ---------------------------------------------------------------------------


def test_longmemeval_target_is_a_target():
    """`build_target` returns something that satisfies the Target
    protocol, and its components satisfy the sub-protocols."""
    t = build_target(eval_backend=MockEval())
    assert isinstance(t, LongMemEvalTarget)
    assert isinstance(t, Target)
    assert isinstance(t.eval_backend, EvalBackend)
    assert isinstance(t.action_space, ActionSpace)
    assert isinstance(t.taxonomy, RegimeTaxonomy)
    assert t.name == "longmemeval"


def test_longmemeval_components_are_independently_constructible():
    """Each component can be instantiated on its own — no hidden coupling
    forces them to be built through `build_target`."""
    aspace = LongMemEvalActionSpace()
    tax = LongMemEvalTaxonomy()
    t = LongMemEvalTarget(
        eval_backend=MockEval(), action_space=aspace, taxonomy=tax,
    )
    assert isinstance(t, Target)


def test_taxonomy_REGIMES_matches_module_level():
    """LongMemEvalTaxonomy.REGIMES() must return the same registry the
    detectors operate on (the loop reads optimizable / seam_reachable
    flags off these to choose targets)."""
    from regimes.loop.regimes import REGIMES as module_REGIMES
    tax = LongMemEvalTaxonomy()
    a = tax.REGIMES()
    b = module_REGIMES()
    assert set(a) == set(b)
    for name in a:
        assert a[name].optimizable == b[name].optimizable
        assert a[name].seam_reachable == b[name].seam_reachable


def test_outcome_summary_back_compat_reexport():
    """`regimes.loop.behaviors._outcome_summary` is the symbol tests
    have always imported. Make sure the back-compat re-export still
    points at the LongMemEval implementation."""
    from regimes.loop.behaviors import _outcome_summary
    from regimes.targets.longmemeval.outcome_summary import outcome_summary
    assert _outcome_summary is outcome_summary


# ---------------------------------------------------------------------------
# 2) run_loop() driven through the Target interface
# ---------------------------------------------------------------------------


def test_run_loop_accepts_explicit_target():
    """run_loop(target=..., ...) drives the same chain as the legacy
    eval_backend= form."""
    target = build_target(eval_backend=MockEval())
    rep = run_loop(target=target, instances=_mini_split(),
                   pause_after="histogram")
    assert rep.histogram is not None
    assert rep.baseline is not None
    # Pause-after-histogram path: stopped with the documented reason.
    assert rep.stopped is not None
    assert rep.stopped["reason"] == "pause_after_histogram"


def test_run_loop_target_and_eval_backend_paths_match():
    """Constructing the Target manually and passing it must produce the
    same LoopReport (overall accuracy + regime counts + stop reason)
    as the legacy eval_backend= path. Locks in the no-behavior-change
    invariant for Phase 1."""
    insts = _mini_split()

    rep_legacy = run_loop(eval_backend=MockEval(), instances=insts,
                          pause_after="histogram")
    rep_target = run_loop(
        target=build_target(eval_backend=MockEval()),
        instances=insts, pause_after="histogram",
    )

    assert rep_legacy.baseline["overall_accuracy"] == rep_target.baseline["overall_accuracy"]
    assert rep_legacy.baseline["per_type_accuracy"] == rep_target.baseline["per_type_accuracy"]
    # Histograms equal modulo qid order. Compare counts per regime.
    counts_legacy = {r["regime"]: r["count"] for r in rep_legacy.histogram["rows"]}
    counts_target = {r["regime"]: r["count"] for r in rep_target.histogram["rows"]}
    assert counts_legacy == counts_target
    assert rep_legacy.stopped["reason"] == rep_target.stopped["reason"]


def test_run_loop_rejects_no_target_no_backend():
    """If neither `target` nor `eval_backend` is passed, the runner
    raises rather than silently constructing a half-built loop."""
    import pytest
    with pytest.raises(TypeError, match="target=|eval_backend="):
        run_loop(instances=_mini_split())
