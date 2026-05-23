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

TOP_K = 5      # "ranked near the top" window for assembly-crowding
SIGNAL_GAP_K = 20  # "ranked outside near-top" window for signal-gap


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


def detect_assemble_internal(o: Outcome) -> bool:
    """Gold-session turns made it INTO selected_turn_ids but the verdict
    is still wrong. The retrieval system did its job; whatever produced
    the wrong answer happened downstream of `assemble()` (reader misread
    the context, prompt formatting, judge call). A score-transform can't
    fix this — the relevant content was already in front of the reader.
    """
    return bool(o.gold_selected()) and not o.correct


def detect_budget_truncation(o: Outcome) -> bool:
    """The agent CONSIDERED a gold-session turn and dropped it at the
    budget wall. Most actionable regime: a transform that filters
    likely-irrelevant high-scoring filler frees budget for the gold
    turn. Detected via the explicit decisions log entry."""
    if not o.truncated:
        return False
    return _gold_dropped_at_budget(o)


def detect_assembly_crowding(o: Outcome) -> bool:
    """Gold-session turn was ranked in the top-K but never made it into
    selected_turn_ids. The agent scored it well but didn't select it —
    typically because higher-scoring non-gold turns ate the seed budget
    before gold could be considered (which differs from budget-truncation,
    where gold WAS considered and decisions records the drop)."""
    if not o.answer_session_ids:
        return False
    in_topk = bool(o.gold_ranked_top_k(TOP_K))
    selected = bool(o.gold_selected())
    return in_topk and not selected


def detect_retrieval_signal_gap(o: Outcome) -> bool:
    """Gold turn is in the scores dict (so scoring ran) but ranked
    outside the SIGNAL_GAP_K window. The scoring signal didn't surface
    gold — fixing this needs a better signal, not a score re-weighting.
    """
    if not o.answer_session_ids or not _gold_in_scores(o):
        return False
    return not bool(o.gold_ranked_top_k(SIGNAL_GAP_K))


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
        name="assemble-internal",
        detector=detect_assemble_internal,
        optimizable=False,
        seam_reachable=False,
        description=(
            "Gold was selected into the context but the answer is still "
            "wrong; the issue is downstream of assemble()."
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
            "Gold ranked near top but never selected; non-gold turns "
            "consumed the seed/budget. Transforms can re-weight to "
            "promote gold."
        ),
    ),
    Regime(
        name="retrieval-signal-gap",
        detector=detect_retrieval_signal_gap,
        optimizable=False,
        seam_reachable=False,
        description=(
            "Gold turn scored too low to surface near the top; the "
            "signal itself misses it. Score-transforms can re-weight "
            "but not invent signal — fix is a signal change."
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


# Priority for classify(): scoring-error must rule out first
# (everything downstream sees garbage scores otherwise);
# assemble-internal must rule out next (gold was selected, so it
# definitively isn't a retrieval regime); the actionable optimizable
# regimes follow in specificity order; seam-unreachable retrieval-signal-gap
# comes last because it's the broadest "low-ranked gold" bucket.
PRIORITY: tuple[str, ...] = (
    "scoring-error",
    "assemble-internal",
    "budget-truncation",
    "assembly-crowding",
    "retrieval-signal-gap",
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
