"""Regression guards for Bug D — regime detector miscalibration.

Bug D symptom: the loop classified all 11 baseline failures as
`assemble-internal` and concluded it had nothing to optimize. Three of
those failures were hand-analyzed and KNOWN to be different regimes:

  - gpt4_a1b77f9c: 6 gold-session turns ranked 4,5,6,8,12,16 of ~498
    but only 1 survived into selected_turn_ids. This is
    assembly-crowding — the signal worked, the assembler dropped the
    material. Score-transforms can re-promote.

  - eac54add: the single gold turn ranked 126 of 476 — never scored
    high enough to be seeded. This is retrieval-signal-gap. A signal
    change is needed, not a score-transform.

  - b46e15ed: bimodal — 3 gold turns ranked 2,3,11, one ranked 166.
    Mixed crowding + signal-gap; the dominant pattern is crowding
    because most well-ranked gold is excluded from selected.

Root cause:
  `detect_assemble_internal` was `bool(o.gold_selected()) and not
  o.correct`. ANY gold-session turn in selected_turn_ids fired it.
  In LongMemEval the agent's seed phase reliably picks at least one
  gold-session turn (often a non-evidence turn that happens to rank
  high), so this detector matched every failure regardless of how
  many gold turns were actually dropped. Combined with `TOP_K=5`
  being too narrow for assembly-crowding (gold ranked at 8-16 didn't
  count as "well-ranked"), every actionable regime fell through to
  the assemble-internal catch-all.

  Priority order made it worse — assemble-internal sat at slot 2,
  pre-empting budget-truncation, assembly-crowding, and
  retrieval-signal-gap.

Fix:
  - WELL_RANKED_K = 20 (was TOP_K=5 for crowding, SIGNAL_GAP_K=20
    for signal-gap — consolidated into one window).
  - assemble-internal now requires COVERAGE >= 0.5 of well-ranked
    gold to be in selected_turn_ids (i.e. the agent retrieved AND
    kept the material — failure is reasoning, not assembly).
  - assembly-crowding fires on coverage < 0.5 (well-ranked gold was
    mostly excluded). Mutually exclusive with assemble-internal by
    construction.
  - retrieval-signal-gap is unchanged (no gold in top-K).
  - Priority reordered: scoring-error → budget-truncation →
    assembly-crowding → retrieval-signal-gap → assemble-internal →
    unclassified. assemble-internal is now a NARROW bucket, not the
    pre-emptive catch-all.

These three cases are pinned as regression fixtures so detector
miscalibration can't recur silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from regimes.eval.types import Outcome
from regimes.loop.regimes import (
    ASSEMBLE_COVERAGE_FLOOR,
    WELL_RANKED_K,
    classify,
    histogram,
)


# ---------------------------------------------------------------------------
# Fixture builder — mirrors the structure of a real LongMemEval outcome
# (498 turns total, gold session, ranking by descending score) at the
# rank profile the user reported. By interleaving gold and distractor
# turns at the specified positions we get exactly the rank counts the
# detector reads off `ranked` + `answer_session_ids`.
# ---------------------------------------------------------------------------


@dataclass
class CaseSpec:
    qid: str
    question_type: str
    gold_session_ids: tuple[str, ...]
    gold_positions: tuple[int, ...]    # 0-indexed positions in `ranked`
    total_turns: int
    selected_positions: tuple[int, ...]  # which positions in ranked are selected
    truncated: bool = True
    decisions: tuple[dict, ...] = ()
    expected_regime: str = ""


def _build_outcome(c: CaseSpec) -> Outcome:
    """Synthesize an Outcome whose ranked tuple has gold-session turns
    at exactly `gold_positions`. Selected = the turns at
    `selected_positions` within ranked."""
    ranked: list[str] = []
    g_iter = iter(sorted(c.gold_positions))
    next_gold = next(g_iter, None)
    gold_seq = 0
    for pos in range(c.total_turns):
        if pos == next_gold:
            sid = c.gold_session_ids[gold_seq % len(c.gold_session_ids)]
            ranked.append(f"{sid}#{pos}")
            gold_seq += 1
            next_gold = next(g_iter, None)
        else:
            ranked.append(f"distractor_{pos}#0")
    scores = {t: 1.0 - i / max(1, c.total_turns) for i, t in enumerate(ranked)}
    selected = tuple(ranked[i] for i in c.selected_positions)
    return Outcome(
        question_id=c.qid,
        question_type=c.question_type,
        is_abstention=False,
        answer_session_ids=c.gold_session_ids,
        correct=False,
        scores=scores,
        ranked=tuple(ranked),
        selected_turn_ids=selected,
        truncated=c.truncated,
        decisions=c.decisions,
    )


# ---------------------------------------------------------------------------
# The three pinned cases (numbers from the user's hand-analysis).
# ---------------------------------------------------------------------------


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        qid="gpt4_a1b77f9c",
        question_type="multi-session",
        gold_session_ids=("gold_a1b77",),
        # "gold evidence ranked 4,5,6,8,12,16 of ~498"
        gold_positions=(4, 5, 6, 8, 12, 16),
        total_turns=498,
        # "only 1 of 6 evidence turns survived into the assembled context"
        selected_positions=(4,),     # 1 of the 6 gold turns in ranked
        truncated=True,
        expected_regime="assembly-crowding",
    ),
    CaseSpec(
        qid="eac54add",
        question_type="temporal-reasoning",
        gold_session_ids=("gold_eac54add",),
        # "the single gold evidence turn ranked 126 of 476"
        gold_positions=(126,),
        total_turns=476,
        # "never scored high enough to be seeded" → not in selected
        selected_positions=(),
        truncated=True,
        expected_regime="retrieval-signal-gap",
    ),
    CaseSpec(
        qid="b46e15ed",
        question_type="multi-session",
        gold_session_ids=("gold_b46e15ed",),
        # "some gold turns ranked 2,3,11, one ranked 166"
        gold_positions=(2, 3, 11, 166),
        total_turns=480,
        # "Mixed crowding + signal-gap" — dominantly crowding because
        # most well-ranked gold (2,3,11) is excluded. Only the top
        # ranked one survives.
        selected_positions=(2,),
        truncated=True,
        expected_regime="assembly-crowding",
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.qid)
def test_pinned_case_classifies_correctly(case: CaseSpec):
    """Each of the three hand-analyzed failures must classify into the
    regime the user identified. This test is the canary — if a future
    detector tweak quietly bucks classification, it fails here."""
    o = _build_outcome(case)
    regime = classify(o)
    assert regime.name == case.expected_regime, (
        f"{case.qid}: classified as {regime.name!r}, "
        f"expected {case.expected_regime!r}"
    )


def test_pinned_cases_recover_optimizable_failure():
    """The bug consequence was that the loop concluded "nothing to
    optimize" because every failure was in a seam-unreachable bucket.
    With the recalibrated detectors, at least one of the three
    hand-analyzed cases (gpt4_a1b77f9c) must classify as a
    seam-reachable optimizable regime — restoring loop progress."""
    outs = [_build_outcome(c) for c in CASES]
    rows = histogram(outs)
    by_regime = {r.regime: r.count for r in rows}
    # assembly-crowding is optimizable + seam-reachable.
    assert by_regime["assembly-crowding"] >= 1, (
        f"no failure recovered into assembly-crowding: {by_regime}"
    )
    # And the loop must NOT classify all three as assemble-internal
    # (the original bug shape).
    assert by_regime["assemble-internal"] < len(CASES), (
        f"all {len(CASES)} cases still in assemble-internal — "
        f"detector miscalibration regressed"
    )


# ---------------------------------------------------------------------------
# Detector-level invariants that the new logic must hold.
# ---------------------------------------------------------------------------


def test_assemble_internal_no_longer_fires_on_minority_gold_selected():
    """The headline bug: 1 of 6 well-ranked gold turns selected used
    to classify as assemble-internal (because gold_selected was
    truthy). Now it must classify as assembly-crowding."""
    c = CASES[0]  # gpt4_a1b77f9c
    o = _build_outcome(c)
    assert classify(o).name != "assemble-internal"
    assert classify(o).name == "assembly-crowding"


def test_signal_gap_fires_when_no_gold_in_well_ranked_window():
    """eac54add: a real signal gap. Single gold turn far outside the
    well-ranked window. Detector must produce signal-gap, not
    assemble-internal (which was the original bug)."""
    c = CASES[1]
    o = _build_outcome(c)
    # Gold IS in scores (rank 126 exists in the ranked tuple).
    # Gold is NOT in top-WELL_RANKED_K.
    assert not o.gold_ranked_top_k(WELL_RANKED_K)
    assert classify(o).name == "retrieval-signal-gap"


def test_coverage_threshold_at_floor_resolves_to_assemble_internal():
    """At exactly ASSEMBLE_COVERAGE_FLOOR, classify resolves to
    assemble-internal — "more than half of well-ranked gold was kept"
    reads as "agent did the retrieval; failure is reasoning". The
    boundary needs to be pinned because it's the meaningful difference
    between an optimizable and a non-optimizable regime."""
    # 2 of 4 well-ranked gold selected → coverage 0.5 → assemble-internal.
    c = CaseSpec(
        qid="boundary",
        question_type="multi-session",
        gold_session_ids=("g",),
        gold_positions=(0, 1, 2, 3),
        total_turns=100,
        selected_positions=(0, 1),
        expected_regime="assemble-internal",
    )
    o = _build_outcome(c)
    assert classify(o).name == "assemble-internal"

    # 1 of 4 well-ranked gold selected → coverage 0.25 → assembly-crowding.
    c2 = CaseSpec(
        qid="boundary2",
        question_type="multi-session",
        gold_session_ids=("g",),
        gold_positions=(0, 1, 2, 3),
        total_turns=100,
        selected_positions=(0,),
        expected_regime="assembly-crowding",
    )
    o2 = _build_outcome(c2)
    assert classify(o2).name == "assembly-crowding"


def test_outcome_carries_signals_the_detectors_need():
    """Defensive: the Outcome's dataclass must actually carry every
    field the detectors read. If a future refactor drops one of these
    the detector falls back silently to weaker logic — pin them."""
    o = _build_outcome(CASES[0])
    # Required signals (per the user's enumeration):
    assert hasattr(o, "answer_session_ids")    # gold identity
    assert hasattr(o, "ranked")                # gold turn ranks
    assert hasattr(o, "scores")                # gold turn scores
    assert hasattr(o, "selected_turn_ids")     # what made it through
    assert hasattr(o, "truncated")             # budget signal
    assert hasattr(o, "decisions")             # per-turn drop reasons
    # And the helpers must be wired:
    assert o.gold_ranked_top_k(WELL_RANKED_K)
    assert o.gold_max_score() > 0.0


def test_priority_reorder_assembly_crowding_beats_assemble_internal():
    """The reorder pins assembly-crowding ahead of assemble-internal in
    priority. With the new mutually-exclusive coverage logic this
    can't matter in practice — but the priority order is the
    documented contract, so we test it explicitly."""
    from regimes.loop.regimes import PRIORITY
    crowding_idx = PRIORITY.index("assembly-crowding")
    internal_idx = PRIORITY.index("assemble-internal")
    assert crowding_idx < internal_idx
