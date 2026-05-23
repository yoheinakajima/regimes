"""The Outcome and EvalResult dataclasses — the single contract between
eval backends (real, mock) and the loop's diagnose step.

The Outcome shape is DELIBERATELY RICH because diagnose has to be able
to classify failures into regimes without re-running the agent:

  Seam-reachable regimes need:
    - assembly-crowding: ranked + selected_turn_ids + answer_session_ids
                         + truncated  (gold ranked high but evicted at
                         the budget wall)
    - budget-truncation: decisions + truncated  (turns dropped at the wall)

  Seam-UNreachable regimes need:
    - retrieval-signal-gap: scores + answer_session_ids  (gold never scored)
    - assemble-internal:    selected_turn_ids + answer_session_ids
                            (gold was selected but assembly hid it)

So Outcome carries the full assembly audit + ranking + post-transform
scores. Diagnose is pure over Outcome — no graph re-walk needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Outcome — what diagnose consumes, one per question
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    # ---- identity ----------------------------------------------------------
    question_id: str
    question_type: str            # canonical: single-session-user, multi-session, ...
    is_abstention: bool           # qid ends with _abs
    answer_session_ids: tuple[str, ...]  # gold sessions; for regime detection

    # ---- judge verdict -----------------------------------------------------
    correct: bool
    judge_label: str = ""         # upstream label: "correct"/"wrong"/"abstain"/...
    judge_raw: dict[str, Any] | None = None  # full judgment record for audit

    # ---- agent retrieval output -------------------------------------------
    hypothesis: str = ""
    signal: str = ""              # "lexical" | "embedding"
    selected_turn_ids: tuple[str, ...] = ()
    n_seeds: int = 0
    n_expanded: int = 0
    truncated: bool = False
    running_tokens: int = 0
    decisions: tuple[dict[str, Any], ...] = ()
    scores: dict[str, float] = field(default_factory=dict)  # post-transform
    ranked: tuple[str, ...] = ()  # top-to-bottom turn-id ranking
    applied_transforms: tuple[str, ...] = ()

    # ---- auditability ------------------------------------------------------
    run_id: str = ""
    error: str | None = None      # populated if agent / reader raised
    # Carried from the agent's behavior.failed event when the scoring step
    # itself blew up (e.g. embedder BadRequestError on a too-long input).
    # Distinct from `error` because the agent can crash inside scoring and
    # still emit downstream events with empty scores; diagnose looks here
    # to classify the failure as scoring-error rather than a retrieval
    # regime.
    score_error: str = ""

    # The specific evidence turn_ids within the gold sessions, as
    # marked by the LongMemEval dataset (`has_answer: true` on each
    # haystack turn, or the top-level `answer_evidences` list).
    # Detectors use these when present to distinguish "the evidence
    # turn was retrieved/selected/dropped" from "any non-evidence turn
    # from the gold session was retrieved/selected/dropped". When the
    # dataset doesn't mark evidence at turn level (e.g. the synthetic
    # fixture), this stays empty and detectors fall back to
    # session-level helpers.
    gold_evidence_turn_ids: tuple[str, ...] = ()

    # ---- derived helpers (cheap, no I/O) ----------------------------------

    def gold_selected(self) -> tuple[str, ...]:
        """Selected turn_ids whose session is in the gold answer set."""
        gold = set(self.answer_session_ids)
        out = []
        for tid in self.selected_turn_ids:
            # turn_id shape is "<session_id>#<turn_idx>"
            sid = tid.split("#", 1)[0] if "#" in tid else tid
            if sid in gold:
                out.append(tid)
        return tuple(out)

    def gold_ranked_top_k(self, k: int) -> tuple[str, ...]:
        """Gold-session turns within the top-k of the ranking. Used by
        assembly-crowding detection."""
        gold = set(self.answer_session_ids)
        out = []
        for tid in self.ranked[:k]:
            sid = tid.split("#", 1)[0] if "#" in tid else tid
            if sid in gold:
                out.append(tid)
        return tuple(out)

    # ---- evidence-turn-level helpers (prefer these over session-level) ----
    #
    # The session-level helpers above conflate "the actual evidence
    # turn was retrieved" with "any non-evidence turn from the same
    # session was retrieved". In LongMemEval the gold session usually
    # has multiple turns, only a subset of which are evidence; the
    # agent reliably picks at least one non-evidence turn that ranks
    # high, which makes session-level detectors classify every failure
    # into whichever regime they prioritize first.
    #
    # When `gold_evidence_turn_ids` is populated, these evidence-level
    # helpers reason about specific turns. The detectors prefer these
    # and fall back to session-level only when evidence-turn IDs are
    # absent (e.g. the synthetic fixture).

    def has_evidence_turn_ids(self) -> bool:
        return bool(self.gold_evidence_turn_ids)

    def evidence_selected(self) -> tuple[str, ...]:
        """The evidence turns that made it into selected_turn_ids."""
        if not self.gold_evidence_turn_ids:
            return ()
        sel = set(self.selected_turn_ids)
        return tuple(t for t in self.gold_evidence_turn_ids if t in sel)

    def evidence_ranked_top_k(self, k: int) -> tuple[str, ...]:
        """Evidence turns appearing in the top-k of the ranking."""
        if not self.gold_evidence_turn_ids:
            return ()
        top = set(self.ranked[:k])
        return tuple(t for t in self.gold_evidence_turn_ids if t in top)

    def evidence_in_scores(self) -> bool:
        """At least one evidence turn appears in the scores dict."""
        if not self.gold_evidence_turn_ids:
            return False
        return any(t in self.scores for t in self.gold_evidence_turn_ids)

    def evidence_dropped_at_budget(self) -> tuple[str, ...]:
        """Evidence turns the agent considered and dropped at the budget
        wall — i.e. they appear in `decisions` with included=False and
        reason='budget'. Used by budget-truncation detection at
        evidence-turn granularity (so a high-scoring NON-evidence turn
        from the gold session being dropped at the budget doesn't get
        misread as the evidence being dropped)."""
        if not self.gold_evidence_turn_ids:
            return ()
        evid = set(self.gold_evidence_turn_ids)
        out = []
        for d in self.decisions:
            tid = str(d.get("turn_id", ""))
            if (
                tid in evid
                and not d.get("included", False)
                and d.get("reason") == "budget"
            ):
                out.append(tid)
        return tuple(out)

    def evidence_max_score(self) -> float:
        """Highest post-transform score on any evidence turn."""
        if not self.gold_evidence_turn_ids:
            return 0.0
        best = 0.0
        for tid in self.gold_evidence_turn_ids:
            s = self.scores.get(tid)
            if s is not None and s > best:
                best = s
        return best

    def evidence_rank_positions(self) -> dict[str, int]:
        """{evidence_turn_id -> 0-indexed rank in `ranked`}. Missing if
        not present in ranked. Used for diagnostics / reporting."""
        if not self.gold_evidence_turn_ids:
            return {}
        rank_of = {tid: i for i, tid in enumerate(self.ranked)}
        return {t: rank_of[t] for t in self.gold_evidence_turn_ids if t in rank_of}

    # ---- session-level helpers (legacy fallback when no evidence-turn IDs) -

    def gold_max_score(self) -> float:
        """Highest post-transform score given to any gold-session turn.
        0.0 if gold never appeared in scores (or has no gold sessions).
        Used by retrieval-signal-gap detection."""
        if not self.answer_session_ids:
            return 0.0
        gold = set(self.answer_session_ids)
        best = 0.0
        for tid, s in self.scores.items():
            sid = tid.split("#", 1)[0] if "#" in tid else tid
            if sid in gold and s > best:
                best = s
        return best


# ---------------------------------------------------------------------------
# EvalResult — what a backend returns
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    outcomes: list[Outcome]
    aggregate: dict[str, Any]   # overall_accuracy, per_type_accuracy, mean tokens, n_truncated, ...
    backend: str                # "real" | "mock"
    run_dir: str | None = None  # path to artifacts; None for in-memory backends
    config: dict[str, Any] = field(default_factory=dict)  # signal, token_budget, applied_transforms, ...

    # ---- summary helpers (used by diagnose + reports) ----

    def overall_accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        n = sum(1 for o in self.outcomes if o.correct)
        return n / len(self.outcomes)

    def per_type_accuracy(self) -> dict[str, float]:
        from collections import defaultdict
        wins: dict[str, int] = defaultdict(int)
        totals: dict[str, int] = defaultdict(int)
        for o in self.outcomes:
            totals[o.question_type] += 1
            if o.correct:
                wins[o.question_type] += 1
        return {t: wins[t] / totals[t] for t in sorted(totals)}

    def mean_running_tokens(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.running_tokens for o in self.outcomes) / len(self.outcomes)

    def by_question_id(self) -> dict[str, Outcome]:
        return {o.question_id: o for o in self.outcomes}


# ---------------------------------------------------------------------------
# Reader / Judge protocols (so production swaps cleanly with fakes)
# ---------------------------------------------------------------------------


@runtime_checkable
class Reader(Protocol):
    """Generates a hypothesis from (context_text, question). Real
    implementation is AnthropicReader (claude-sonnet-4-5, T=0,
    tool-free); test implementation is FakeReader."""

    name: str

    def answer(self, *, context: str, question: str, question_id: str) -> str: ...


@runtime_checkable
class Judge(Protocol):
    """Judges a set of hypotheses against references. Real implementation
    is LMEJudge (shells out to LME's upstream evaluate_qa.py); test
    implementation is FakeJudge."""

    name: str

    def judge(
        self,
        *,
        hypotheses_path: str,
        references_path: str,
        run_dir: str,
    ) -> list[dict[str, Any]]:
        """Return per-question judgments. Each record is at minimum:
            {"question_id": str, "correct": bool, "label": str}
        plus any backend-specific fields preserved into Outcome.judge_raw.
        """
        ...
