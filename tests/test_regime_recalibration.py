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
    # Positions (0-indexed in `ranked`) of EVIDENCE turns — the actual
    # has_answer=true turns the dataset marks. Detectors reason about
    # these, NOT just any turn from the gold session.
    evidence_positions: tuple[int, ...]
    # Positions of NON-evidence gold-session turns (high-scoring filler
    # from the same session). These exist in real LME data: the gold
    # session contains both evidence turns and conversational filler.
    # The detector must see through them — that's the core fix.
    non_evidence_gold_positions: tuple[int, ...]
    total_turns: int
    selected_positions: tuple[int, ...]  # which positions in ranked are selected
    truncated: bool = True
    decisions: tuple[dict, ...] = ()
    expected_regime: str = ""


def _build_outcome(c: CaseSpec) -> Outcome:
    """Synthesize an Outcome whose ranked tuple has evidence turns at
    `evidence_positions` and non-evidence gold-session turns at
    `non_evidence_gold_positions`. Evidence turns get marker IDs that
    are listed on `gold_evidence_turn_ids`; non-evidence gold-session
    turns share the gold session ID but are NOT in evidence."""
    sid = c.gold_session_ids[0]
    ranked: list[str] = []
    evidence_tids: list[str] = []
    ev_set = set(c.evidence_positions)
    ne_set = set(c.non_evidence_gold_positions)
    for pos in range(c.total_turns):
        if pos in ev_set:
            tid = f"{sid}#evidence{pos}"
            ranked.append(tid)
            evidence_tids.append(tid)
        elif pos in ne_set:
            ranked.append(f"{sid}#filler{pos}")
        else:
            ranked.append(f"distractor_{pos}#0")
    scores = {t: 1.0 - i / max(1, c.total_turns) for i, t in enumerate(ranked)}
    selected = tuple(ranked[i] for i in c.selected_positions)
    return Outcome(
        question_id=c.qid,
        question_type=c.question_type,
        is_abstention=False,
        answer_session_ids=c.gold_session_ids,
        gold_evidence_turn_ids=tuple(evidence_tids),
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
        # "gold evidence ranked 4,5,6,8,12,16 of ~498" — all six are
        # actual evidence turns; no non-evidence filler complicates
        # this case.
        evidence_positions=(4, 5, 6, 8, 12, 16),
        non_evidence_gold_positions=(),
        total_turns=498,
        # "only 1 of 6 evidence turns survived into the assembled context"
        selected_positions=(4,),     # the evidence turn at rank 4
        truncated=True,
        expected_regime="assembly-crowding",
    ),
    CaseSpec(
        qid="eac54add",
        question_type="temporal-reasoning",
        gold_session_ids=("gold_eac54add",),
        # "the single gold evidence turn ranked 126 of 476"
        evidence_positions=(126,),
        # The gold session also contains high-scoring non-evidence
        # turns at ranks 2,3,11 — this is the realistic LME profile
        # that broke session-level reasoning. The agent selects these
        # high-scoring non-evidence turns and the detector falsely
        # reads "gold was retrieved well", classifying as
        # budget-truncation or assemble-internal. Evidence-level
        # reasoning must see through this and classify as signal-gap.
        non_evidence_gold_positions=(2, 3, 11),
        total_turns=476,
        # Agent selects the non-evidence filler at positions 2,3.
        # The actual evidence (position 126) is never seeded.
        selected_positions=(2, 3),
        truncated=True,
        # And the budget log even shows one non-evidence gold-session
        # turn dropped at the budget wall — session-level
        # _gold_dropped_at_budget would falsely fire here.
        decisions=(
            {"turn_id": "gold_eac54add#filler11", "included": False,
             "reason": "budget"},
        ),
        expected_regime="retrieval-signal-gap",
    ),
    CaseSpec(
        qid="b46e15ed",
        question_type="multi-session",
        gold_session_ids=("gold_b46e15ed",),
        # "some gold turns ranked 2,3,11, one ranked 166" — all four
        # are evidence turns (bimodal evidence distribution).
        evidence_positions=(2, 3, 11, 166),
        non_evidence_gold_positions=(),
        total_turns=480,
        # "Mixed crowding + signal-gap" — dominantly crowding because
        # most well-ranked evidence (2,3,11) is excluded. Only the
        # top-ranked one survives.
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
    """eac54add: a real signal gap. The evidence turn is far outside
    the well-ranked window even though non-evidence turns from the
    gold session score high. Detector must use EVIDENCE-level
    granularity to reach signal-gap."""
    c = CASES[1]
    o = _build_outcome(c)
    # Session-level helper sees the filler turns at ranks 2,3,11 —
    # would falsely indicate "gold well-ranked".
    assert o.gold_ranked_top_k(WELL_RANKED_K), (
        "fixture sanity: session-level helper should see filler "
        "turns at high ranks (this is the session/evidence conflation "
        "that the detector must see through)"
    )
    # Evidence-level helper correctly returns empty — the actual
    # evidence at rank 126 is not in the well-ranked window.
    assert not o.evidence_ranked_top_k(WELL_RANKED_K)
    # Therefore the detector classifies as signal-gap, not crowding
    # or assemble-internal.
    assert classify(o).name == "retrieval-signal-gap"


def test_coverage_threshold_at_floor_resolves_to_assemble_internal():
    """At exactly ASSEMBLE_COVERAGE_FLOOR, classify resolves to
    assemble-internal — "more than half of well-ranked gold was kept"
    reads as "agent did the retrieval; failure is reasoning". The
    boundary needs to be pinned because it's the meaningful difference
    between an optimizable and a non-optimizable regime."""
    # 2 of 4 well-ranked evidence selected → coverage 0.5 → assemble-internal.
    c = CaseSpec(
        qid="boundary",
        question_type="multi-session",
        gold_session_ids=("g",),
        evidence_positions=(0, 1, 2, 3),
        non_evidence_gold_positions=(),
        total_turns=100,
        selected_positions=(0, 1),
        expected_regime="assemble-internal",
    )
    o = _build_outcome(c)
    assert classify(o).name == "assemble-internal"

    # 1 of 4 well-ranked evidence selected → coverage 0.25 → assembly-crowding.
    c2 = CaseSpec(
        qid="boundary2",
        question_type="multi-session",
        gold_session_ids=("g",),
        evidence_positions=(0, 1, 2, 3),
        non_evidence_gold_positions=(),
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


# ---------------------------------------------------------------------------
# Evidence-level vs session-level — the killer test for the second
# round of detector bugs.
# ---------------------------------------------------------------------------


def test_eac54add_does_not_misclassify_as_budget_truncation():
    """Session-level reasoning falsely classified eac54add as
    budget-truncation because the gold SESSION had non-evidence turns
    ranked high enough to be considered AND one was logged in decisions
    with reason='budget'. The actual evidence turn (rank 126) was
    never even seeded. With evidence-turn granularity the detector
    must see signal-gap, not budget-truncation."""
    case = next(c for c in CASES if c.qid == "eac54add")
    o = _build_outcome(case)
    regime = classify(o)
    assert regime.name != "budget-truncation", (
        "eac54add classified as budget-truncation — the session-level "
        "conflation regressed (non-evidence gold-session turns being "
        "dropped at budget falsely matches budget-truncation)."
    )
    assert regime.name == "retrieval-signal-gap"


def test_evidence_level_signals_carry_through_outcome():
    """The Outcome must carry per-evidence-turn signals — not just
    session-level. Pin the required helpers/fields here so a future
    schema change can't silently drop them and re-introduce the
    session-vs-evidence conflation."""
    case = next(c for c in CASES if c.qid == "eac54add")
    o = _build_outcome(case)
    assert hasattr(o, "gold_evidence_turn_ids")
    assert o.has_evidence_turn_ids()
    # Each evidence helper must exist and return evidence-specific data.
    assert hasattr(o, "evidence_ranked_top_k")
    assert hasattr(o, "evidence_selected")
    assert hasattr(o, "evidence_in_scores")
    assert hasattr(o, "evidence_dropped_at_budget")
    # And they must produce the right values for the eac54add profile:
    # - evidence at rank 126 → not in top-20
    assert o.evidence_ranked_top_k(20) == ()
    # - evidence not selected (only non-evidence filler was)
    assert o.evidence_selected() == ()
    # - evidence IS in scores (gold was scored, just not surfaced)
    assert o.evidence_in_scores()
    # - the budget-drop log records a non-evidence turn, not the
    #   evidence turn itself
    assert o.evidence_dropped_at_budget() == ()


# ---------------------------------------------------------------------------
# Non-uniformity guard: the user's 11 real failures must NOT all land
# in one regime. Acceptance test for "detector is discriminating".
# ---------------------------------------------------------------------------


def _build_eleven_realistic_failures() -> list[Outcome]:
    """Synthesize 11 failures mirroring the rank/selection profiles
    of the user's actual baseline run. Per their hand-analysis:
      - 1 signal-gap (eac54add profile: evidence ranked >100)
      - several crowding (gpt4_a1b77f9c profile: evidence in top-K,
        most excluded under truncation)
      - a few assemble-internal (evidence in top-K AND mostly
        selected, but reader still wrong)
      - the remainder mixed.

    The exact split doesn't matter for the guard — what matters is
    that the histogram spans at least two regimes."""
    extras: list[Outcome] = []
    # The three pinned cases first.
    for c in CASES:
        extras.append(_build_outcome(c))
    # Two more in signal-gap shape (evidence buried far down,
    # non-evidence gold-session turns ranking high).
    for i, qid in enumerate(("sig_gap_q1", "sig_gap_q2")):
        extras.append(_build_outcome(CaseSpec(
            qid=qid, question_type="multi-session",
            gold_session_ids=(f"gs_{i}",),
            evidence_positions=(200 + i,),
            non_evidence_gold_positions=(1, 4, 7),
            total_turns=400,
            selected_positions=(1, 4),
            truncated=True,
        )))
    # Two more crowding (evidence well-ranked, mostly excluded).
    for i, qid in enumerate(("crowd_q1", "crowd_q2")):
        extras.append(_build_outcome(CaseSpec(
            qid=qid, question_type="multi-session",
            gold_session_ids=(f"gc_{i}",),
            evidence_positions=(3, 4, 9, 14),
            non_evidence_gold_positions=(),
            total_turns=300,
            selected_positions=(3,),
            truncated=True,
        )))
    # Two assemble-internal (evidence mostly selected, answer wrong).
    for i, qid in enumerate(("ai_q1", "ai_q2")):
        extras.append(_build_outcome(CaseSpec(
            qid=qid, question_type="multi-session",
            gold_session_ids=(f"ga_{i}",),
            evidence_positions=(0, 1, 2),
            non_evidence_gold_positions=(),
            total_turns=200,
            selected_positions=(0, 1, 2),
            truncated=False,
        )))
    # Two budget-truncation (evidence well-ranked, in decisions with
    # reason='budget' explicitly).
    for i, qid in enumerate(("bt_q1", "bt_q2")):
        extras.append(_build_outcome(CaseSpec(
            qid=qid, question_type="multi-session",
            gold_session_ids=(f"gb_{i}",),
            evidence_positions=(2, 4),
            non_evidence_gold_positions=(),
            total_turns=300,
            selected_positions=(2,),
            truncated=True,
            decisions=(
                {"turn_id": f"gb_{i}#evidence4", "included": False,
                 "reason": "budget"},
            ),
        )))
    return extras[:11]


def test_eleven_failure_distribution_spans_multiple_regimes():
    """Acceptance test: the recalibrated detectors must produce a
    non-uniform distribution on a realistic mix of failures. The
    previous two bugs (assemble-internal catch-all, budget-truncation
    catch-all) both presented as uniform 11/11 distribution into one
    bucket. We assert the histogram spans ≥2 regimes.

    The stronger acceptance — ideally signal-gap + crowding both
    present — is also asserted, since those are the two seam-relevant
    regimes the loop's transform action space depends on
    distinguishing."""
    outs = _build_eleven_realistic_failures()
    assert len(outs) == 11
    rows = histogram(outs)
    by_regime = {r.regime: r.count for r in rows if r.count > 0}

    # Non-uniformity: at least 2 distinct buckets occupied.
    assert len(by_regime) >= 2, (
        f"11 failures collapsed into a single bucket: {by_regime}. "
        f"This is the symptom shape of detector non-discrimination."
    )

    # And no single regime owns more than 80% — a softer check that
    # catches "almost everything bucketed in one regime" near-misses.
    max_share = max(by_regime.values()) / len(outs)
    assert max_share <= 0.80, (
        f"one regime owns {max_share:.0%} of failures: {by_regime}. "
        f"Detector is barely discriminating."
    )

    # Both signal-gap and crowding (or budget-truncation) present —
    # these are the two regimes the loop's action space differentiates
    # treatment for. The detector must surface both when the data has
    # both.
    assert by_regime.get("retrieval-signal-gap", 0) >= 1, (
        f"signal-gap missing despite signal-gap fixtures: {by_regime}"
    )
    crowding = (
        by_regime.get("assembly-crowding", 0)
        + by_regime.get("budget-truncation", 0)
    )
    assert crowding >= 1, (
        f"no crowding/budget-truncation despite crowding fixtures: "
        f"{by_regime}"
    )
