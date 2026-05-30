"""End-to-end proof that the reader-prompt seam works AND that the loop
rotates through every seam-reachable regime before stopping.

Two gaps these tests pin:

  Gap 1 (rotation). With budget-truncation the dominant optimizable
  regime and assemble-internal also seam-reachable, the loop must
  ATTEMPT BOTH: it drafts score-transforms for budget-truncation, and
  after that regime is exhausted (max_consecutive_discards discards) it
  must ROTATE to assemble-internal and draft a reader-prompt-transform —
  not stop the whole loop.

  Gap 3 (reader-prompt application). A drafted reader-prompt-transform
  must actually be APPLIED to the reader's prompt during eval-diff such
  that it changes the eval outcome. The mock reader (`MockReader`)
  re-reads the modified prompt_parts; when the transform injects the
  reconciliation instruction the verdict flips correct, the
  assemble-internal regime shrinks, and the transform promotes.

Before the fix, a reader-prompt-transform installed into a pipeline no
eval path ever read, so its eval-diff was always a no-op and it was
always discarded — the promotion assertions below would fail.
"""

from __future__ import annotations

import pytest

from regimes.loop import (
    TRANSFORM_DRAFTED,
    TRANSFORM_PROMOTED,
    run_loop,
)
from regimes.loop.mock_eval import MockEval, MockInstance, MockReader
from regimes.targets.longmemeval.action_space import clear_all_pipelines
from regimes.targets.longmemeval.mock_author import MockTypedAuthor

# The reconciliation text the mock reader-prompt author injects into the
# `instruction` fragment (see mock_author._READER_LIBRARY). The mock
# reader treats its presence as "the reader now answers correctly".
RECONCILE_MARKER = "prefer the most recent entry"


@pytest.fixture(autouse=True)
def _clean():
    clear_all_pipelines()
    yield
    clear_all_pipelines()


# ---------------------------------------------------------------------------
# Fixtures: a baseline with budget-truncation dominant + assemble-internal.
# ---------------------------------------------------------------------------


def _budget_truncation_instance(i: int) -> MockInstance:
    """Well-ranked gold dropped at the budget wall → budget-truncation.

    No gold_score_threshold, so the drafted score-transform can never
    flip it correct → it is discarded, exhausting the regime."""
    return MockInstance(
        f"q_bt{i}", "multi-session", False, (f"sGb{i}",), False,
        scores={f"sGb{i}#0": 0.7, f"sNb{i}#0": 0.6},
        ranked=(f"sGb{i}#0", f"sNb{i}#0"),
        selected_turn_ids=(f"sNb{i}#0",), truncated=True,
        decisions=({"turn_id": f"sGb{i}#0", "included": False, "reason": "budget"},),
        candidate_turn_ids=(f"sGb{i}#0", f"sNb{i}#0"),
    )


def _assemble_internal_instance(i: int) -> MockInstance:
    """Well-ranked gold, fully selected (coverage 1.0), answer wrong, not
    truncated → assemble-internal. Carries reader prompt fragments + the
    marker so a reader-prompt-transform that injects RECONCILE_MARKER
    flips it correct."""
    gold = f"sGa{i}"
    return MockInstance(
        f"q_ai{i}", "multi-session", False, (gold,), False,
        scores={f"{gold}#0": 0.9, f"sNa{i}#0": 0.4},
        ranked=(f"{gold}#0", f"sNa{i}#0"),
        selected_turn_ids=(f"{gold}#0",),
        truncated=False,
        prompt_parts=(
            ("instruction", "Answer the question based on the context."),
            ("context", f"turn {gold}#0"),
        ),
        reader_correct_when_contains=RECONCILE_MARKER,
    )


def _baseline_mix() -> list[MockInstance]:
    """6 budget-truncation + 5 assemble-internal + 1 correct — the real
    run's baseline shape (budget-truncation dominant, both reachable)."""
    insts: list[MockInstance] = [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}, selected_turn_ids=("s_ok#0",)),
    ]
    insts += [_budget_truncation_instance(i) for i in range(6)]
    insts += [_assemble_internal_instance(i) for i in range(5)]
    return insts


# ---------------------------------------------------------------------------
# Gap 1: rotation — the loop attempts BOTH regimes before stopping.
# ---------------------------------------------------------------------------


def test_loop_rotates_to_assemble_internal_after_budget_truncation_exhausted():
    rep = run_loop(
        eval_backend=MockEval(),
        instances=_baseline_mix(),
        author=MockTypedAuthor(),
        max_consecutive_discards=3,
    )

    # Histogram still flags assemble-internal reachable (the derived flags).
    rows = {r["regime"]: r for r in rep.histogram["rows"]}
    assert rows["budget-truncation"]["count"] == 6, rows
    assert rows["assemble-internal"]["count"] == 5, rows
    assert rows["assemble-internal"]["optimizable"] is True
    assert rows["assemble-internal"]["seam_reachable"] is True

    drafted = [e.payload for e in rep.events if e.type == TRANSFORM_DRAFTED]
    targeted = {d["target_regime"] for d in drafted}
    # BOTH seam-reachable regimes must have been attempted.
    assert "budget-truncation" in targeted, drafted
    assert "assemble-internal" in targeted, drafted

    # budget-truncation drafted score-transforms; assemble-internal a
    # reader-prompt-transform — the seam routing is visible in the log.
    types_by_regime = {d["target_regime"]: d["transform_type"] for d in drafted}
    assert types_by_regime["budget-truncation"] in (
        "score-transform", "assembly-transform",
    )
    assert types_by_regime["assemble-internal"] == "reader-prompt-transform"

    # budget-truncation is exhausted (drafted up to max_consecutive_discards
    # times) BEFORE the rotation to assemble-internal.
    n_budget_drafts = sum(
        1 for d in drafted if d["target_regime"] == "budget-truncation"
    )
    assert n_budget_drafts == 3, drafted
    # And the rotation reached assemble-internal afterwards.
    regimes_in_order = [d["target_regime"] for d in drafted]
    assert regimes_in_order.index("assemble-internal") > regimes_in_order.index(
        "budget-truncation"
    )


# ---------------------------------------------------------------------------
# Gap 2: transform_type is visible on every transform_log entry.
# ---------------------------------------------------------------------------


def test_transform_log_records_transform_type_for_each_seam():
    rep = run_loop(
        eval_backend=MockEval(),
        instances=_baseline_mix(),
        author=MockTypedAuthor(),
        max_consecutive_discards=3,
    )
    assert rep.transform_log
    # Every entry exposes a readable seam (never "?").
    for rec in rep.transform_log:
        assert rec.get("transform_type") in (
            "score-transform", "assembly-transform", "reader-prompt-transform",
        ), rec
    seams = {rec["transform_type"] for rec in rep.transform_log}
    assert "reader-prompt-transform" in seams, rep.transform_log


# ---------------------------------------------------------------------------
# Gap 3: the reader-prompt-transform drafts → gates → APPLIES → flips
# outcome → promotes. The full end-to-end proof.
# ---------------------------------------------------------------------------


def test_reader_prompt_transform_applies_to_reader_and_changes_outcome():
    rep = run_loop(
        eval_backend=MockEval(),
        instances=_baseline_mix(),
        author=MockTypedAuthor(),
        max_consecutive_discards=3,
    )

    # (a)+(b) rotated to assemble-internal and drafted a reader-prompt-transform.
    drafted = [e.payload for e in rep.events if e.type == TRANSFORM_DRAFTED]
    reader_drafts = [
        d for d in drafted if d["transform_type"] == "reader-prompt-transform"
    ]
    assert reader_drafts, "loop never drafted a reader-prompt-transform"
    assert reader_drafts[0]["target_regime"] == "assemble-internal"

    # (c)+(d) it passed gates AND its application flipped the outcome, so it
    # PROMOTED — only possible if the modified prompt reached the reader and
    # shrank the assemble-internal regime.
    promoted = [e.payload for e in rep.events if e.type == TRANSFORM_PROMOTED]
    reader_promotions = [
        rec for rec in rep.transform_log
        if rec["transform_type"] == "reader-prompt-transform"
        and rec["status"] == "promoted"
    ]
    assert reader_promotions, (
        "reader-prompt-transform never promoted — it did not change the "
        f"eval outcome. transform_log={rep.transform_log}"
    )
    rp = reader_promotions[0]
    assert rp["target_regime"] == "assemble-internal"
    # Targeted regime shrank (negative target_delta) and overall improved.
    assert rp["target_delta"] < 0, rp
    assert rp["overall_delta"] > 0, rp
    assert any(p["target_regime"] == "assemble-internal" for p in promoted)


def test_mock_reader_sees_modified_prompt_parts_directly():
    """Unit-level proof of the seam the loop relies on: with a
    reader-prompt-transform installed, MockEval's reader reads the
    MODIFIED prompt_parts and flips the verdict; reverting restores it."""
    from regimes.agent import reader_transforms

    inst = _assemble_internal_instance(0)
    ev = MockEval()

    # Baseline: no transform installed → instruction lacks the marker →
    # the mock reader leaves the wrong answer wrong.
    base = ev.run_on_split([inst])
    assert base.outcomes[0].correct is False

    # Install the reader-prompt-transform the mock author would draft.
    src = (
        "def transform(prompt_parts, question, question_date):\n"
        "    out = dict(prompt_parts)\n"
        "    out['instruction'] = out.get('instruction', '') + "
        "' When evidence conflicts, prefer the most recent entry.'\n"
        "    return out\n"
    )
    ns: dict = {}
    exec(compile(src, "<t>", "exec"), ns, ns)  # noqa: S102 — test fixture
    reader_transforms.promote("t_reader", ns["transform"])
    try:
        after = ev.run_on_split([inst])
        # The modified prompt reached the reader and flipped the verdict.
        assert after.outcomes[0].correct is True
        assert "t_reader" in after.outcomes[0].applied_transforms
    finally:
        reader_transforms.revert("t_reader")

    # Reverted → back to wrong.
    restored = ev.run_on_split([inst])
    assert restored.outcomes[0].correct is False


def test_mock_reader_marker_absent_keeps_baseline():
    """Sanity: a reader-prompt-transform that does NOT inject the marker
    leaves the verdict unchanged (the seam isn't a free pass)."""
    reader = MockReader()
    assert reader.reads_correct(
        prompt_parts={"instruction": "Answer.", "context": "x"},
        required_marker=RECONCILE_MARKER,
        baseline_correct=False,
    ) is False
