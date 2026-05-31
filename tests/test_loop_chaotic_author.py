"""Rotation-under-chaos tests: the loop's iteration state machine must be
robust to EVERY candidate outcome, not just the clean ones MockTypedAuthor
produces.

The real LLM author emits malformed code, catastrophic regressions, and
mixed outcomes; the mock author here (ChaoticMockAuthor) reproduces that
mix so the static_rejected / sandbox_rejected control-flow branches are
actually exercised.

The pinned bug (gap 4): a regime whose attempts ended discard, discard,
static_rejected fell out of the iteration with `stopped: None` — no
rotation to the next seam-reachable regime, no clean stop. Before the
fix, TRANSFORM_STATIC_REJECTED and TRANSFORM_SANDBOX_REJECTED had NO
listener, so the chain simply died mid-iteration and the counter never
advanced.

Invariants under test:
  * static_rejected, sandbox_rejected, discarded and confirm_regression
    ALL count as failed attempts toward max_consecutive_discards.
  * at the ceiling, the regime is retired and the loop ROTATES to the
    next un-exhausted seam-reachable regime.
  * a completed run ALWAYS emits a non-None stopped block (stopped: None
    is impossible).
"""

from __future__ import annotations

import pytest

from regimes.eval.types import EvalResult
from regimes.loop import (
    LOOP_STOPPED,
    TRANSFORM_DISCARDED,
    TRANSFORM_DRAFTED,
    TRANSFORM_PROMOTED,
    TRANSFORM_SANDBOX_REJECTED,
    TRANSFORM_STATIC_REJECTED,
    MockEval,
    MockInstance,
    run_loop,
)
from regimes.targets.longmemeval.action_space import clear_all_pipelines
from regimes.targets.longmemeval.mock_author import (
    RECONCILE_MARKER,
    ChaoticMockAuthor,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_all_pipelines()
    yield
    clear_all_pipelines()


# ---------------------------------------------------------------------------
# Fixtures: budget-truncation dominant + assemble-internal both reachable.
# ---------------------------------------------------------------------------


def _budget_truncation_instance(i: int) -> MockInstance:
    """Well-ranked gold dropped at the budget wall → budget-truncation.
    No flip path (no gold_score_threshold), so any score-transform the
    chaotic author drafts cannot promote it."""
    return MockInstance(
        f"q_bt{i}", "multi-session", False, (f"sGb{i}",), False,
        scores={f"sGb{i}#0": 0.7, f"sNb{i}#0": 0.6},
        ranked=(f"sGb{i}#0", f"sNb{i}#0"),
        selected_turn_ids=(f"sNb{i}#0",), truncated=True,
        decisions=({"turn_id": f"sGb{i}#0", "included": False, "reason": "budget"},),
        candidate_turn_ids=(f"sGb{i}#0", f"sNb{i}#0"),
    )


def _assemble_internal_instance(i: int) -> MockInstance:
    """Well-ranked gold, fully selected, answer wrong, not truncated →
    assemble-internal. Carries prompt fragments + the reconciliation
    marker so a reader-prompt "promote" candidate flips it correct."""
    gold = f"sGa{i}"
    return MockInstance(
        f"q_ai{i}", "multi-session", False, (gold,), False,
        scores={f"{gold}#0": 0.9, f"sNa{i}#0": 0.4},
        ranked=(f"{gold}#0", f"sNa{i}#0"),
        selected_turn_ids=(f"{gold}#0",), truncated=False,
        prompt_parts=(
            ("instruction", "Answer the question based on the context."),
            ("context", f"turn {gold}#0"),
        ),
        reader_correct_when_contains=RECONCILE_MARKER,
    )


def _baseline_mix() -> list[MockInstance]:
    insts: list[MockInstance] = [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}, selected_turn_ids=("s_ok#0",)),
    ]
    insts += [_budget_truncation_instance(i) for i in range(6)]
    insts += [_assemble_internal_instance(i) for i in range(5)]
    return insts


def _drafted(rep) -> list[dict]:
    return [e.payload for e in rep.events if e.type == TRANSFORM_DRAFTED]


def _statuses(rep) -> list[str]:
    return [r["status"] for r in rep.transform_log]


# ---------------------------------------------------------------------------
# Test 1 (the exact real failure): discard, discard, static_rejected →
# rotate to the next seam-reachable regime, NOT exit with stopped=None.
# ---------------------------------------------------------------------------


def test_discard_discard_static_reject_rotates_not_stops_none():
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["discard", "discard", "static_reject"],
        "assemble-internal": ["promote"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=3,
    )

    # The bug: this used to be None. It must NEVER be None on a completed run.
    assert rep.stopped is not None, "loop exited with stopped: None (gap 4)"

    # budget-truncation got exactly 3 attempts (discard, discard,
    # static_rejected) — the static_rejected counted toward the ceiling
    # instead of falling out of the iteration.
    statuses = _statuses(rep)
    assert statuses[:3] == ["discarded", "discarded", "static_rejected"], statuses

    # And the loop ROTATED to assemble-internal afterwards.
    targeted = [d["target_regime"] for d in _drafted(rep)]
    assert "budget-truncation" in targeted
    assert "assemble-internal" in targeted, targeted
    assert targeted.index("assemble-internal") > targeted.index("budget-truncation")
    n_budget = sum(1 for t in targeted if t == "budget-truncation")
    assert n_budget == 3, targeted

    # A real static_rejected event was emitted (the branch is exercised).
    assert any(e.type == TRANSFORM_STATIC_REJECTED for e in rep.events)


# ---------------------------------------------------------------------------
# Test 2: a regime whose author ONLY writes garbage (all static_rejected)
# still counts as exhausted and rotates.
# ---------------------------------------------------------------------------


def test_all_static_reject_exhausts_and_rotates():
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["static_reject"],  # repeats forever
        "assemble-internal": ["promote"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=3,
    )

    assert rep.stopped is not None
    statuses = _statuses(rep)
    # 3 static_rejected for budget-truncation (exhausts), then it rotated.
    assert statuses[:3] == ["static_rejected"] * 3, statuses
    targeted = [d["target_regime"] for d in _drafted(rep)]
    assert sum(1 for t in targeted if t == "budget-truncation") == 3, targeted
    assert "assemble-internal" in targeted, targeted


def test_all_sandbox_reject_exhausts_and_rotates():
    """The sandbox-reject branch is symmetric to static-reject: a crashing
    candidate must also count toward the ceiling and rotate."""
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["sandbox_reject"],
        "assemble-internal": ["promote"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=3,
    )
    assert rep.stopped is not None
    statuses = _statuses(rep)
    assert statuses[:3] == ["sandbox_rejected"] * 3, statuses
    assert any(e.type == TRANSFORM_SANDBOX_REJECTED for e in rep.events)
    targeted = [d["target_regime"] for d in _drafted(rep)]
    assert "assemble-internal" in targeted, targeted


# ---------------------------------------------------------------------------
# Test 3: mixed across two regimes — budget-truncation exhausts via a
# discard/reject mix → rotates to assemble-internal → drafts a
# reader-prompt-transform → it applies to the reader and PROMOTES → loop
# ends with a proper stopped block listing what was attempted.
# ---------------------------------------------------------------------------


def test_mixed_rejects_rotate_to_reader_and_promote():
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["discard", "sandbox_reject", "static_reject"],
        "assemble-internal": ["promote"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=3,
    )

    # Proper stop block (not None) with a named reason.
    assert rep.stopped is not None
    assert rep.stopped["reason"] in (
        "no_optimizable_regime_remaining", "max_iterations",
    ), rep.stopped

    # budget-truncation exercised all three failure branches, in order.
    statuses = _statuses(rep)
    assert statuses[:3] == ["discarded", "sandbox_rejected", "static_rejected"], statuses

    # The full failure variety is represented in the audit log.
    distinct = {r["status"] for r in rep.transform_log}
    assert {"discarded", "sandbox_rejected", "static_rejected"} <= distinct

    # Rotation drafted a reader-prompt-transform for assemble-internal...
    reader_drafts = [
        d for d in _drafted(rep)
        if d["transform_type"] == "reader-prompt-transform"
    ]
    assert reader_drafts, _drafted(rep)
    assert reader_drafts[0]["target_regime"] == "assemble-internal"

    # ...and it APPLIED to the reader and promoted (shrank the regime).
    reader_promotions = [
        r for r in rep.transform_log
        if r["transform_type"] == "reader-prompt-transform"
        and r["status"] == "promoted"
    ]
    assert reader_promotions, rep.transform_log
    assert reader_promotions[0]["target_delta"] < 0
    assert any(p["target_regime"] == "assemble-internal" for p in rep.promotions)


def test_mixed_reader_can_be_confirm_discarded():
    """The other branch of test 3's 'promote OR confirm-discard': the
    reader-prompt-transform improves OPTIMIZE but a confirm-regressing
    backend discards it with reason='confirm_regression'. Still ends with
    a non-None stopped block."""
    import dataclasses

    from regimes.agent import reader_transforms as _RT

    class _RegressReaderOnConfirm:
        """Flips marked confirm instances wrong whenever a reader-prompt
        transform is installed — simulating overfit on OPTIMIZE."""

        def __init__(self, regress_qids):
            self._base = MockEval()
            self._regress = frozenset(regress_qids)

        def run_on_split(self, instances, **kw):
            result = self._base.run_on_split(instances, **kw)
            if not _RT.get_pipeline():
                return result
            new = []
            for o in result.outcomes:
                if o.question_id in self._regress:
                    new.append(dataclasses.replace(o, correct=False, judge_label="mock-0"))
                else:
                    new.append(o)
            return EvalResult(
                outcomes=new, aggregate=result.aggregate, backend=result.backend,
                run_dir=result.run_dir, config=result.config,
            )

    confirm = [
        MockInstance("qc_ok1", "multi-session", False, ("sc1",), True,
                     scores={"sc1#0": 1.0}, selected_turn_ids=("sc1#0",)),
        MockInstance("qc_ok2", "multi-session", False, ("sc2",), True,
                     scores={"sc2#0": 0.9}, selected_turn_ids=("sc2#0",)),
    ]
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["static_reject"],
        "assemble-internal": ["promote"],
    })
    backend = _RegressReaderOnConfirm(regress_qids={"qc_ok1", "qc_ok2"})
    rep = run_loop(
        eval_backend=backend, instances=_baseline_mix(),
        confirm_instances=confirm, author=author, max_consecutive_discards=3,
    )
    assert rep.stopped is not None
    # The reader candidate was confirm-discarded, not promoted.
    assert TRANSFORM_PROMOTED not in {e.type for e in rep.events}
    discards = [d for d in rep.discards if "confirm_regression" in d.get("reasons", [])]
    assert discards, rep.transform_log


# ---------------------------------------------------------------------------
# Test 4: EVERY completed chaotic run emits a non-None stopped block.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget_script", [
    ["discard", "discard", "static_reject"],
    ["static_reject"],
    ["sandbox_reject"],
    ["discard", "sandbox_reject", "static_reject"],
    ["static_reject", "discard", "sandbox_reject"],
    ["discard"],
    ["sandbox_reject", "static_reject", "discard"],
])
def test_every_completed_chaotic_run_emits_nonnull_stopped(budget_script):
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": budget_script,
        "assemble-internal": ["discard"],   # even the rotation target fails
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=3,
    )
    assert rep.stopped is not None, (
        f"stopped: None for budget_script={budget_script}"
    )
    # Exactly one terminal stop event, and it carries a reason.
    stop_events = [e for e in rep.events if e.type == LOOP_STOPPED]
    assert len(stop_events) == 1, [e.type for e in rep.events]
    assert rep.stopped["reason"], rep.stopped
    # The backstop synthetic reason must never be the cause — the behaviors
    # themselves must always emit the stop.
    assert rep.stopped["reason"] != "loop_drained_without_stop", rep.stopped


def test_both_regimes_exhaust_then_clean_stop():
    """When BOTH seam-reachable regimes are exhausted by garbage, the loop
    stops once (no rotation target left), listing what remains."""
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["static_reject"],
        "assemble-internal": ["discard"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=2,
    )
    targeted = [d["target_regime"] for d in _drafted(rep)]
    # Both attempted (2 each at ceiling=2) before the stop.
    assert sum(1 for t in targeted if t == "budget-truncation") == 2, targeted
    assert sum(1 for t in targeted if t == "assemble-internal") == 2, targeted
    assert rep.stopped is not None
    assert rep.stopped["reason"] == "no_optimizable_regime_remaining"


# ---------------------------------------------------------------------------
# Per-outcome counting: each non-promoting outcome counts as ONE failed
# attempt (no double-count, no under-count).
# ---------------------------------------------------------------------------


def test_static_reject_counts_as_single_failed_attempt():
    """ceiling=1 → the FIRST static_reject must exhaust budget-truncation
    and rotate immediately (proves the bump happens, exactly once)."""
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["static_reject"],
        "assemble-internal": ["promote"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=1,
    )
    targeted = [d["target_regime"] for d in _drafted(rep)]
    assert sum(1 for t in targeted if t == "budget-truncation") == 1, targeted
    assert "assemble-internal" in targeted, targeted


def test_discard_counts_are_not_double_bumped():
    """ceiling=2 → exactly 2 discards before rotation (the centralized bump
    in _handle_failed_attempt must not double-count with the old in-promote
    increment)."""
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["discard"],
        "assemble-internal": ["promote"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=2,
    )
    targeted = [d["target_regime"] for d in _drafted(rep)]
    assert sum(1 for t in targeted if t == "budget-truncation") == 2, targeted


# ---------------------------------------------------------------------------
# A clean-author run is unchanged (no regression of the happy path).
# ---------------------------------------------------------------------------


def test_clean_author_still_promotes_and_stops():
    author = ChaoticMockAuthor(by_regime={
        "budget-truncation": ["discard"],
        "assemble-internal": ["promote"],
    })
    rep = run_loop(
        eval_backend=MockEval(), instances=_baseline_mix(),
        author=author, max_consecutive_discards=1,
    )
    assert rep.stopped is not None
    assert any(p["target_regime"] == "assemble-internal" for p in rep.promotions)
