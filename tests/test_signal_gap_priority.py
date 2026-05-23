"""Regression guards for Bug F — never-retrieved cases leaking into the
optimizable bucket.

After Bug E the loop persisted only qid/correct/regime per failure, and
the eac54add case (1 evidence turn answer_0d4d0348_2#0, hand-found at
rank 126 in an earlier run) was classified as `budget-truncation`.
budget-truncation means "gold was selected then dropped at the budget
wall" — but at rank 126 the evidence was never effectively seeded, so
this label was wrong AND it leaked a never-retrieved case into the
optimizable bucket the loop would have wasted eval budget trying to
fix with score-transforms.

Root cause: detect_budget_truncation only required `truncated=True`
AND `_gold_dropped_at_budget` (any gold turn in decisions with
reason='budget'). It didn't check whether the evidence was actually
WELL-RANKED to begin with. The agent's assembly iterates ranked turns
with score > 0 — even rank-126 turns get a budget-drop trail in
decisions.

Fix: budget-truncation now requires `evidence_ranked_top_k(WELL_RANKED_K)`
non-empty. assembly-crowding already had this requirement via the
coverage helper. Plus retrieval-signal-gap moved earlier in PRIORITY
so any future detector loosening still can't accidentally classify a
never-retrieved case as recoverable.

These tests pin both the eac54add classification AND the structural
invariant: no outcome whose evidence is outside the well-ranked top-K
can EVER classify as budget-truncation or assembly-crowding.
"""

from __future__ import annotations

import pytest

from regimes.eval.types import Outcome
from regimes.loop.regimes import (
    WELL_RANKED_K,
    classify,
    detect_assembly_crowding,
    detect_budget_truncation,
    detect_retrieval_signal_gap,
)


def _build_eac54add_at_rank_126() -> Outcome:
    """eac54add's reported profile: 1 evidence turn at rank 126
    (answer_0d4d0348_2#0), agent's selection took higher-ranked
    non-evidence content, and the budget log records the evidence
    turn being dropped at the budget wall (because the agent iterated
    that far through the score order)."""
    evidence_tid = "answer_0d4d0348_2#0"
    total = 476
    ranked: list[str] = []
    for pos in range(total):
        if pos == 126:
            ranked.append(evidence_tid)
        else:
            ranked.append(f"distractor_{pos}#0")
    scores = {t: 1.0 - i / total for i, t in enumerate(ranked)}
    # Selected: 15 top-scoring non-evidence turns. The evidence turn
    # was iterated past and logged in decisions with reason='budget'.
    selected = tuple(ranked[i] for i in range(15) if ranked[i] != evidence_tid)
    decisions = (
        {"turn_id": evidence_tid, "included": False, "reason": "budget"},
    )
    return Outcome(
        question_id="eac54add",
        question_type="temporal-reasoning",
        is_abstention=False,
        answer_session_ids=("answer_0d4d0348_2",),
        gold_evidence_turn_ids=(evidence_tid,),
        correct=False,
        scores=scores,
        ranked=tuple(ranked),
        selected_turn_ids=selected,
        truncated=True,
        decisions=decisions,
    )


def test_eac54add_classifies_as_signal_gap_not_budget_truncation():
    """The headline regression: with evidence at rank 126 (outside
    WELL_RANKED_K) and dropped at the budget wall, the detector must
    NOT call this budget-truncation. The eval budget the loop would
    spend trying to optimize a never-retrieved case is wasted."""
    o = _build_eac54add_at_rank_126()
    regime = classify(o)
    assert regime.name != "budget-truncation", (
        "eac54add classified as budget-truncation — Bug F regressed. "
        "Evidence at rank 126 is not effectively seeded; transforms "
        "can't recover it. This must be signal-gap."
    )
    assert regime.name == "retrieval-signal-gap"


def test_eac54add_signals_match_the_label():
    """The label must be derivable from the persisted signals — that's
    the whole point of the signal-bearing report. Pin each signal
    eac54add carries so a future serialization change can't drop
    them silently."""
    o = _build_eac54add_at_rank_126()
    # 1) The evidence turn exists, and is in scores (so it WAS scored).
    assert o.has_evidence_turn_ids()
    assert o.evidence_in_scores()
    # 2) The evidence rank is 126 — OUTSIDE the well-ranked window.
    ranks = o.evidence_rank_positions()
    assert ranks == {"answer_0d4d0348_2#0": 126}
    assert not o.evidence_ranked_top_k(WELL_RANKED_K), (
        "evidence at rank 126 must not appear in top-WELL_RANKED_K"
    )
    # 3) Evidence is NOT in selected (it was never seeded effectively).
    assert o.evidence_selected() == ()
    # 4) The budget-drop trail is real — and that's exactly the trap
    #    the old detector fell into.
    assert o.evidence_dropped_at_budget() == ("answer_0d4d0348_2#0",)
    # The label this combination of signals justifies is signal-gap.
    # ("Evidence in scores but never well-ranked.")
    assert classify(o).name == "retrieval-signal-gap"


# ---------------------------------------------------------------------------
# Structural invariant: never-retrieved cases CANNOT classify as either
# optimizable regime. This is the contract the loop's transform-search
# economics depend on — no eval budget should ever be spent on a case
# the action space provably can't reach.
# ---------------------------------------------------------------------------


def _outcome_with_evidence_at(rank: int, total: int = 500, *,
                              evidence_in_decisions_as_budget: bool = False,
                              evidence_selected_too: bool = False) -> Outcome:
    """Build an outcome with one evidence turn placed at `rank`."""
    ranked: list[str] = []
    for pos in range(total):
        if pos == rank:
            ranked.append("ev_session#0")
        else:
            ranked.append(f"distractor_{pos}#0")
    scores = {t: 1.0 - i / max(1, total) for i, t in enumerate(ranked)}
    decisions = ()
    if evidence_in_decisions_as_budget:
        decisions = (
            {"turn_id": "ev_session#0", "included": False, "reason": "budget"},
        )
    selected = ("ev_session#0",) if evidence_selected_too else ()
    return Outcome(
        question_id=f"q_evidence_rank_{rank}",
        question_type="multi-session",
        is_abstention=False,
        answer_session_ids=("ev_session",),
        gold_evidence_turn_ids=("ev_session#0",),
        correct=False,
        scores=scores,
        ranked=tuple(ranked),
        selected_turn_ids=selected,
        truncated=True,
        decisions=decisions,
    )


@pytest.mark.parametrize("rank", [WELL_RANKED_K, WELL_RANKED_K + 1, 50, 126, 300])
def test_never_retrieved_evidence_cannot_be_budget_truncation(rank: int):
    """For evidence ranked outside the well-ranked window — even with
    the decisions log marking the evidence as dropped at budget — the
    classification MUST NOT be budget-truncation. The optimizable
    bucket is reserved for cases score-transforms can actually reach."""
    o = _outcome_with_evidence_at(rank, evidence_in_decisions_as_budget=True)
    assert not detect_budget_truncation(o), (
        f"evidence at rank {rank} classified as budget-truncation. "
        f"This is the never-retrieved leak from Bug F — a score-"
        f"transform can't move a rank-{rank} turn into the seed window."
    )


@pytest.mark.parametrize("rank", [WELL_RANKED_K, WELL_RANKED_K + 1, 50, 126, 300])
def test_never_retrieved_evidence_cannot_be_assembly_crowding(rank: int):
    """Same invariant for crowding: cannot fire when no evidence is
    well-ranked. assembly-crowding's `_well_ranked_gold_coverage`
    helper already enforces this; this test pins the invariant."""
    o = _outcome_with_evidence_at(rank, evidence_in_decisions_as_budget=True)
    assert not detect_assembly_crowding(o), (
        f"evidence at rank {rank} classified as assembly-crowding"
    )


@pytest.mark.parametrize("rank", [WELL_RANKED_K, 21, 50, 126, 300])
def test_never_retrieved_evidence_classifies_as_signal_gap(rank: int):
    """Mirror of the two negative invariants above — these cases
    SHOULD classify as signal-gap. Asserting both negative-and-positive
    pins the boundary."""
    o = _outcome_with_evidence_at(rank, evidence_in_decisions_as_budget=True)
    assert detect_retrieval_signal_gap(o)
    assert classify(o).name == "retrieval-signal-gap"


def test_evidence_at_top_k_boundary_is_well_ranked():
    """Sanity-check the boundary. Evidence at rank WELL_RANKED_K - 1
    (the last position INSIDE the window) IS well-ranked. With the
    evidence dropped at budget, this is genuine budget-truncation."""
    o = _outcome_with_evidence_at(
        WELL_RANKED_K - 1,
        evidence_in_decisions_as_budget=True,
    )
    assert detect_budget_truncation(o)
    assert classify(o).name == "budget-truncation"


# ---------------------------------------------------------------------------
# Self-justifying report: signals must persist into outcome summaries.
# ---------------------------------------------------------------------------


def test_outcome_summary_carries_evidence_signals():
    """The loop's persisted summary (per behavior_run_baseline and
    behavior_rebaseline) must carry the evidence-level signals that
    drove the regime label. Without this every label in the report
    is unauditable — the original Bug F symptom."""
    from regimes.loop.behaviors import _outcome_summary

    o = _build_eac54add_at_rank_126()
    summary = _outcome_summary(o)

    # The regime label is present AND classifies correctly.
    assert summary["regime"] == "retrieval-signal-gap"

    # Every detector-input signal must be on the summary so the label
    # can be checked against its basis.
    required_signals = {
        "gold_evidence_turn_ids",
        "evidence_rank_positions",
        "evidence_in_scores",
        "evidence_max_score",
        "evidence_well_ranked",
        "evidence_selected",
        "evidence_dropped_at_budget",
        "evidence_coverage",
        "well_ranked_k",
    }
    missing = required_signals - set(summary)
    assert not missing, (
        f"persisted summary missing detector-input signals: {missing}"
    )

    # And the values must reflect what the detector actually read.
    assert summary["gold_evidence_turn_ids"] == ["answer_0d4d0348_2#0"]
    assert summary["evidence_rank_positions"] == {"answer_0d4d0348_2#0": 126}
    assert summary["evidence_well_ranked"] == []
    assert summary["evidence_selected"] == []
    assert summary["evidence_dropped_at_budget"] == ["answer_0d4d0348_2#0"]
    assert summary["evidence_coverage"] is None  # no well-ranked → no coverage
    assert summary["evidence_in_scores"] is True
    assert summary["well_ranked_k"] == WELL_RANKED_K


def test_outcome_summary_is_json_serializable():
    """The summary lives in event payloads and the report.json file —
    every value must round-trip JSON cleanly. Floats, lists, dicts,
    bools, ints, None and str only — no tuples / NaN / etc."""
    import json

    from regimes.loop.behaviors import _outcome_summary

    o = _build_eac54add_at_rank_126()
    summary = _outcome_summary(o)
    # Round-trip:
    text = json.dumps(summary)
    back = json.loads(text)
    assert back["regime"] == "retrieval-signal-gap"
    assert back["evidence_rank_positions"] == {"answer_0d4d0348_2#0": 126}


def test_outcome_summary_priority_order_pins_signal_gap_first():
    """Defense-in-depth invariant: PRIORITY tuple lists
    retrieval-signal-gap before budget-truncation and
    assembly-crowding. Any future detector loosening will still get
    short-circuited by signal-gap when evidence isn't well-ranked."""
    from regimes.loop.regimes import PRIORITY

    sg = PRIORITY.index("retrieval-signal-gap")
    bt = PRIORITY.index("budget-truncation")
    ac = PRIORITY.index("assembly-crowding")
    assert sg < bt, "signal-gap must come before budget-truncation"
    assert sg < ac, "signal-gap must come before assembly-crowding"
