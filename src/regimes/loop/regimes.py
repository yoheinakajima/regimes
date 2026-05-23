"""Regime taxonomy + deterministic detectors.

The diagnose phase classifies each failing Outcome into one regime. The
taxonomy is fixed at the top of this module; each entry pairs a name
with a deterministic detector function. Detectors are PURE over Outcome
— no graph re-walk, no I/O — so they're replayable and trivially
testable.

The taxonomy splits two ways:
  - optimizable vs. non-optimizable
        scoring-error is non-optimizable: there's no score to re-weight
        when the scoring step itself failed.
  - seam-reachable vs. seam-unreachable
        score-transforms operate at the turns.scored seam. Regimes
        whose fix lives outside that seam (signal change, assemble()
        internals, scoring bug) are unreachable; the loop's `stop`
        phase names them as walls instead of trying to optimize them.

Priority order for `classify()` is important when an Outcome's signals
match more than one detector — see PRIORITY below. The order picks the
most-specific / most-actionable regime first.

The LLM-proposed-new-regime hook is `register_regime(name, detector,
*, optimizable, seam_reachable)`. Once registered the detector is pure
replayable code; the loop never invokes the LLM to RE-classify.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from regimes.eval.types import Outcome


# ----- Threshold knobs -----------------------------------------------------
#
# These are part of the detector contract. Bumping them changes diagnose
# output and therefore the histogram — treat as a versioned schema.
#
# WELL_RANKED_K: the rank window we consider "the agent retrieved gold
# successfully". 20 matches the order of magnitude where the assembler
# can still seed a gold turn under a 2500-token budget; gold ranked
# outside this window never gets considered by the seed phase.
#
# ASSEMBLE_COVERAGE_FLOOR: the fraction of well-ranked gold turns that
# must be in selected_turn_ids before we call it assemble-internal
# (i.e. "the agent got the retrieval right; failure is downstream").
# Coverage below this floor with well-ranked gold present means the
# assembler/budget dropped material the score signal had surfaced —
# assembly-crowding (refined to budget-truncation when the decisions
# log records the explicit budget drop).
#
# The earlier detectors used TOP_K=5 (too narrow — gold at ranks 8-16
# wasn't "well-ranked") and called any gold-session turn in selected
# "assemble-internal" without measuring coverage. That collapsed every
# real failure into the assemble-internal catch-all because the agent
# usually picks at least one gold-session turn (often a non-evidence
# turn that happens to rank high), and verdicts were wrong, so the
# binary check matched.

WELL_RANKED_K = 20
ASSEMBLE_COVERAGE_FLOOR = 0.5

# Kept as legacy aliases for any external callers / tests that import
# them; the detectors below use WELL_RANKED_K throughout.
TOP_K = WELL_RANKED_K
SIGNAL_GAP_K = WELL_RANKED_K


# ----- Regime descriptor ---------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """A regime descriptor: a name + a binary detector + classification
    flags. Detectors take an Outcome and return True iff the outcome's
    failure matches this regime's signature."""

    name: str
    detector: Callable[[Outcome], bool]
    optimizable: bool       # can a score-transform plausibly fix it?
    seam_reachable: bool    # is the fix at the turns.scored seam?
    description: str = ""


# ----- Detector helpers ----------------------------------------------------


def _gold_sids(o: Outcome) -> set[str]:
    return set(o.answer_session_ids)


def _gold_in_scores(o: Outcome) -> bool:
    """At least one turn from a gold session appears in the scores dict.
    Score value (incl. zero) doesn't matter for this check — what
    matters is whether the scoring step *produced* an entry."""
    gold = _gold_sids(o)
    if not gold:
        return False
    for tid in o.scores:
        sid = tid.split("#", 1)[0] if "#" in tid else tid
        if sid in gold:
            return True
    return False


def _gold_dropped_at_budget(o: Outcome) -> bool:
    """A gold-session turn appears in `decisions` with included=False
    and reason='budget'. The agent considered it but the budget wall
    killed the include."""
    gold = _gold_sids(o)
    if not gold:
        return False
    for d in o.decisions:
        tid = str(d.get("turn_id", ""))
        sid = tid.split("#", 1)[0] if "#" in tid else tid
        if sid in gold and not d.get("included", False) and d.get("reason") == "budget":
            return True
    return False


# ----- Built-in detectors --------------------------------------------------


def detect_scoring_error(o: Outcome) -> bool:
    """The scoring step did not produce usable scores for this question.

    Two sub-cases — both are non-optimizable by a score-transform:

      a) The agent's score behavior raised. Surfaced via
         Outcome.score_error (lifted from the trace's behavior.failed
         event by agent.retrieve()).

      b) Scoring completed but gold-session turns are entirely absent
         from the produced scores dict. score_lexical and
         score_embedding are dense by contract; missing-from-dict means
         either the scoring math skipped them or the ingest didn't put
         the gold sessions in the corpus. Either way the score-transform
         seam can't help.
    """
    if o.score_error:
        return True
    if o.answer_session_ids and not _gold_in_scores(o):
        return True
    return False


def _well_ranked_gold_coverage(o: Outcome) -> tuple[int, int]:
    """Return (n_well_ranked_gold_selected, n_well_ranked_gold_total).

    "Well-ranked gold" = gold-session turns appearing in the top
    WELL_RANKED_K of `ranked`. "Selected" = also appearing in
    selected_turn_ids. The pair drives the assembly-crowding /
    assemble-internal split: coverage = selected / total, computed
    against the well-ranked gold turns (NOT the whole gold set —
    a gold turn ranked at position 200 was never seed-eligible and
    its non-inclusion isn't an assembly failure)."""
    well_ranked = set(o.gold_ranked_top_k(WELL_RANKED_K))
    if not well_ranked:
        return (0, 0)
    selected_set = set(o.selected_turn_ids)
    n_selected = len(well_ranked & selected_set)
    return (n_selected, len(well_ranked))


def detect_assemble_internal(o: Outcome) -> bool:
    """The agent retrieved gold WELL — gold turns ranked in the top
    WELL_RANKED_K AND most of them made it into selected_turn_ids —
    but the answer is still wrong. The failure is downstream of
    assemble() (reader misread the context, prompt formatting, judge
    call). A score-transform cannot fix this.

    The previous implementation was `bool(o.gold_selected()) and not
    o.correct` — any gold-session turn in selected_turn_ids would
    match. That conflated "the assembler kept the gold material"
    with "the assembler kept any turn from the gold session", and
    bucketed every real failure (assembly-crowding, budget-truncation,
    signal-gap) into this catch-all whenever the agent's seed phase
    happened to pick even one gold-session turn.
    """
    if not o.answer_session_ids or not _gold_in_scores(o):
        return False
    n_sel, n_total = _well_ranked_gold_coverage(o)
    if n_total == 0:
        return False  # no well-ranked gold → signal-gap territory
    return (n_sel / n_total) >= ASSEMBLE_COVERAGE_FLOOR and not o.correct


def detect_budget_truncation(o: Outcome) -> bool:
    """The agent CONSIDERED a gold-session turn and dropped it at the
    budget wall. Most actionable regime: a transform that filters
    likely-irrelevant high-scoring filler frees budget for the gold
    turn. Detected via the explicit decisions log entry."""
    if not o.truncated:
        return False
    return _gold_dropped_at_budget(o)


def detect_assembly_crowding(o: Outcome) -> bool:
    """Gold turns were ranked WELL by the signal (in top WELL_RANKED_K)
    but most were NOT selected into the assembled context. The signal
    did its job; the assembler/budget evicted material the loop's
    score-transform action space can re-promote.

    Coverage threshold: with well-ranked gold present, fewer than
    ASSEMBLE_COVERAGE_FLOOR of them made it through to
    selected_turn_ids. (At exactly the floor, assemble-internal wins —
    "more than half of well-ranked gold was kept" reads as "agent did
    the retrieval, reasoning is the failure".)
    """
    if not o.answer_session_ids or not _gold_in_scores(o):
        return False
    n_sel, n_total = _well_ranked_gold_coverage(o)
    if n_total == 0:
        return False  # gold never ranked well → signal-gap territory
    return (n_sel / n_total) < ASSEMBLE_COVERAGE_FLOOR


def detect_retrieval_signal_gap(o: Outcome) -> bool:
    """Gold turn is in the scores dict (so scoring ran) but NO gold
    turn ranked within the top WELL_RANKED_K. The scoring signal
    didn't surface gold; a score-transform can re-weight existing
    scores but can't invent signal — fixing this needs a signal
    change (different embedder / additional scorer / etc.)."""
    if not o.answer_session_ids or not _gold_in_scores(o):
        return False
    return not bool(o.gold_ranked_top_k(WELL_RANKED_K))


def detect_unclassified(o: Outcome) -> bool:  # noqa: ARG001
    """Fall-through. Always True; only ever reached if every preceding
    detector returned False — see PRIORITY below."""
    return True


# ----- The fixed taxonomy --------------------------------------------------


_BUILTIN: list[Regime] = [
    Regime(
        name="scoring-error",
        detector=detect_scoring_error,
        optimizable=False,
        seam_reachable=False,
        description=(
            "Scoring step failed or produced no entries for gold turns; "
            "non-optimizable by a score-transform."
        ),
    ),
    Regime(
        name="budget-truncation",
        detector=detect_budget_truncation,
        optimizable=True,
        seam_reachable=True,
        description=(
            "Gold turn was considered and dropped at the budget wall — "
            "transforms can demote filler to free budget for it."
        ),
    ),
    Regime(
        name="assembly-crowding",
        detector=detect_assembly_crowding,
        optimizable=True,
        seam_reachable=True,
        description=(
            "Gold ranked well but most well-ranked gold turns were "
            "excluded from selected_turn_ids — fewer than "
            "ASSEMBLE_COVERAGE_FLOOR made it through. Transforms can "
            "re-weight scores to promote gold past competing filler."
        ),
    ),
    Regime(
        name="retrieval-signal-gap",
        detector=detect_retrieval_signal_gap,
        optimizable=False,
        seam_reachable=False,
        description=(
            "No gold turn ranked within the top WELL_RANKED_K; the "
            "signal itself misses gold. Score-transforms can re-weight "
            "existing scores but not invent signal — fix is a signal "
            "change."
        ),
    ),
    Regime(
        name="assemble-internal",
        detector=detect_assemble_internal,
        optimizable=False,
        seam_reachable=False,
        description=(
            "Well-ranked gold was MOSTLY selected into the context "
            "(coverage >= ASSEMBLE_COVERAGE_FLOOR) but the answer is "
            "still wrong. Failure is downstream of assemble() — reader, "
            "prompt format, judge — not addressable by a score-transform."
        ),
    ),
    Regime(
        name="unclassified",
        detector=detect_unclassified,
        optimizable=False,
        seam_reachable=False,
        description=(
            "No built-in detector matched; flagged for LLM inspection "
            "to propose a new named regime."
        ),
    ),
]


# Priority for classify():
#   1. scoring-error first — without scores there's no retrieval regime
#      to classify against.
#   2. budget-truncation next when the agent EXPLICITLY recorded a gold
#      turn dropped at the budget wall (most specific actionable
#      signal; refines crowding).
#   3. assembly-crowding when well-ranked gold was mostly excluded
#      (poor coverage).
#   4. retrieval-signal-gap when no gold turn ranked well (mutually
#      exclusive with crowding by construction — crowding requires
#      well-ranked gold).
#   5. assemble-internal only when retrieval AND assembly succeeded
#      (coverage >= floor) and the answer is still wrong. This is now
#      a NARROW bucket, not the catch-all it used to be.
#   6. unclassified — anything left.
#
# The previous order ran assemble-internal at slot 2, which let its
# permissive "any gold turn selected" detector pre-empt every
# actionable regime. The new order is also the order detectors should
# be specified in PER-OUTCOME mutually-exclusive logic:
# scoring-error → budget vs not → coverage low vs high → no well-ranked → other.
PRIORITY: tuple[str, ...] = (
    "scoring-error",
    "budget-truncation",
    "assembly-crowding",
    "retrieval-signal-gap",
    "assemble-internal",
    "unclassified",
)


# ----- Mutable registry (LLM-proposed regimes append here) -----------------

_REG_LOCK = threading.Lock()
_REGISTRY: dict[str, Regime] = {r.name: r for r in _BUILTIN}
_PRIORITY: list[str] = list(PRIORITY)


def REGIMES() -> dict[str, Regime]:  # noqa: N802 — public-API style
    """Snapshot of the current regime registry."""
    with _REG_LOCK:
        return dict(_REGISTRY)


def register_regime(
    name: str,
    detector: Callable[[Outcome], bool],
    *,
    optimizable: bool,
    seam_reachable: bool,
    description: str = "",
    priority_after: str = "assembly-crowding",
) -> None:
    """Append a new regime to the taxonomy. Loop hypothesize callers use
    this when a failure cluster fits no built-in detector. The detector
    is pure replayable code from this point on; the LLM author is never
    re-consulted to classify."""
    with _REG_LOCK:
        if name in _REGISTRY:
            raise ValueError(f"regime already registered: {name!r}")
        _REGISTRY[name] = Regime(
            name=name,
            detector=detector,
            optimizable=optimizable,
            seam_reachable=seam_reachable,
            description=description,
        )
        try:
            idx = _PRIORITY.index(priority_after)
        except ValueError:
            idx = len(_PRIORITY) - 1  # before unclassified
        _PRIORITY.insert(idx + 1, name)


def reset_regimes() -> None:
    """Restore the built-in taxonomy. Test isolation only."""
    with _REG_LOCK:
        _REGISTRY.clear()
        _REGISTRY.update({r.name: r for r in _BUILTIN})
        _PRIORITY[:] = list(PRIORITY)


# ----- Classification + histogram ------------------------------------------


def classify(o: Outcome) -> Regime:
    """Return the highest-priority regime whose detector fires for this
    outcome. Always terminates: `unclassified` matches everything as a
    safety floor."""
    with _REG_LOCK:
        for name in _PRIORITY:
            r = _REGISTRY[name]
            if r.detector(o):
                return r
        # Defensive: unreachable while `unclassified` is in PRIORITY.
        return _REGISTRY["unclassified"]


def is_seam_reachable(regime_name: str) -> bool:
    """Is a regime addressable by the score-transform action space?"""
    with _REG_LOCK:
        r = _REGISTRY.get(regime_name)
    return bool(r and r.seam_reachable)


@dataclass(frozen=True)
class HistogramRow:
    regime: str
    count: int
    optimizable: bool
    seam_reachable: bool
    qids: tuple[str, ...] = field(default_factory=tuple)


def histogram(outcomes: list[Outcome], *, failures_only: bool = True) -> list[HistogramRow]:
    """Count outcomes per regime. Stable order: PRIORITY order, then
    new regimes registered after import.

    failures_only=True (default) counts only `correct == False` outcomes,
    which is what diagnose feeds into the headroom-vs-bug decision.
    failures_only=False counts every outcome — useful for full audit
    when you want to see what regime the correct ones look like too."""
    target = [o for o in outcomes if (not failures_only) or (not o.correct)]
    by_regime: dict[str, list[Outcome]] = {n: [] for n in _PRIORITY}
    for o in target:
        r = classify(o)
        by_regime.setdefault(r.name, []).append(o)

    with _REG_LOCK:
        rows = []
        for name in _PRIORITY:
            r = _REGISTRY[name]
            members = by_regime.get(name, [])
            rows.append(
                HistogramRow(
                    regime=name,
                    count=len(members),
                    optimizable=r.optimizable,
                    seam_reachable=r.seam_reachable,
                    qids=tuple(o.question_id for o in members),
                )
            )
    return rows


def format_histogram(
    rows: list[HistogramRow],
    *,
    n_failures: int,
    n_total: int,
) -> str:
    """Single-line-per-row text report. Used by the runner to print the
    histogram at the pause point."""
    lines = [
        f"Regime histogram (failures={n_failures} / total={n_total}):",
        f"  {'regime':<24s}  {'count':>5s}  {'opt':>4s}  {'seam':>5s}",
    ]
    for r in rows:
        flag_opt = "yes" if r.optimizable else "no"
        flag_seam = "yes" if r.seam_reachable else "no"
        lines.append(
            f"  {r.regime:<24s}  {r.count:>5d}  {flag_opt:>4s}  {flag_seam:>5s}"
        )
    return "\n".join(lines)
