"""MockEval — an in-memory eval backend with controllable per-question
failure modes.

The loop's diagnose / hypothesize / gates code is wired to the same
`Outcome` / `EvalResult` contract as `RealEval`. MockEval produces
those Outcomes deterministically without touching the agent, the LME
checkout, or any network — so the loop can be exercised end-to-end with
no keys.

The mock is fixture-driven: each `MockInstance` specifies the desired
Outcome shape (correct, scores, ranked, selected_turn_ids, decisions,
truncated, score_error) directly. This lets tests construct precise
regime mixes (e.g. "5 assembly-crowding + 3 budget-truncation + 1
scoring-error") and verify diagnose / histogram / gates against them.

Transforms are honored: if a MockInstance carries a `gold_score_under`
threshold and a transform-pipeline is registered (via
regimes.agent.transforms.promote()), the mock will simulate the
transform by re-running the pipeline against the recorded scores and
recomputing correctness with the simple rule "any gold-session turn
crosses the threshold ⇒ correct". This is enough to exercise the
gates' eval-diff math without faking a full retrieval re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regimes.agent import transforms as _transforms
from regimes.eval.types import EvalResult, Outcome


# ----- Instance spec -------------------------------------------------------


@dataclass(frozen=True)
class MockInstance:
    """One synthetic eval question. Field meanings match Outcome's.

    Tests typically construct these directly to pin a regime mix.

    The `gold_score_under` threshold + `correct_when_gold_score_at_least`
    pair lets the mock simulate a transform's effect: after applying the
    registered transform pipeline to `scores`, if any gold-session turn
    ends up with a score >= the threshold AND that turn was in the
    selected_turn_ids candidate list, the answer flips to correct.
    """

    question_id: str
    question_type: str
    is_abstention: bool
    answer_session_ids: tuple[str, ...]
    correct_baseline: bool          # what the baseline (no transform) says
    scores: dict[str, float] = field(default_factory=dict)
    ranked: tuple[str, ...] = ()
    selected_turn_ids: tuple[str, ...] = ()
    decisions: tuple[dict[str, Any], ...] = ()
    truncated: bool = False
    score_error: str = ""
    n_seeds: int = 0
    n_expanded: int = 0
    running_tokens: int = 0
    question: str = ""
    question_date: str = ""
    # Simulator hook for transform-flipping (see module docstring).
    gold_score_threshold: float = float("inf")
    candidate_turn_ids: tuple[str, ...] = ()


# ----- Backend -------------------------------------------------------------


@dataclass
class MockEval:
    """Backend with the same shape as RealEval. `run_on_split(instances,
    run_dir=...)` produces an EvalResult from a list of MockInstance.

    Accepts ordinary dicts too: callers can pass either MockInstance
    objects OR dicts with the same keys, so test fixtures + loop
    scripts share a path."""

    signal: str = "embedding"
    token_budget: int = 2500

    def run_on_split(
        self,
        instances: list[MockInstance | dict[str, Any]],
        *,
        run_dir: str | Path | None = None,
    ) -> EvalResult:
        outs: list[Outcome] = []
        for inst in instances:
            mi = inst if isinstance(inst, MockInstance) else _from_dict(inst)
            outs.append(self._build_outcome(mi))

        from collections import defaultdict

        per_type_correct: dict[str, int] = defaultdict(int)
        per_type_total: dict[str, int] = defaultdict(int)
        n_truncated = 0
        n_errors = 0
        total_tokens = 0
        for o in outs:
            per_type_total[o.question_type] += 1
            if o.correct:
                per_type_correct[o.question_type] += 1
            if o.truncated:
                n_truncated += 1
            if o.error or o.score_error:
                n_errors += 1
            total_tokens += o.running_tokens

        agg = {
            "version": "regimes-mock-v1",
            "n": len(outs),
            "overall_accuracy": (
                sum(1 for o in outs if o.correct) / len(outs) if outs else 0.0
            ),
            "per_type_accuracy": {
                t: per_type_correct[t] / per_type_total[t] for t in sorted(per_type_total)
            },
            "n_truncated": n_truncated,
            "n_errors": n_errors,
            "mean_context_tokens": total_tokens / len(outs) if outs else 0.0,
        }
        return EvalResult(
            outcomes=outs,
            aggregate=agg,
            backend="mock",
            run_dir=(str(run_dir) if run_dir is not None else None),
            config={
                "signal": self.signal,
                "token_budget": self.token_budget,
                "applied_transforms": [e.name for e in _transforms.get_pipeline()],
            },
        )

    # ----- per-instance helpers --------------------------------------------

    def _build_outcome(self, mi: MockInstance) -> Outcome:
        scores = dict(mi.scores)
        applied = []
        # Run the registered transform pipeline against the recorded
        # scores — same call shape as the real agent's seam.
        if scores and _transforms.get_pipeline():
            result, _errors = _transforms.apply_pipeline(
                scores=scores,
                graph=None,             # mock — transforms that need graph won't
                                        # work here; the sandbox gate runs those
                                        # against real graphs.
                question_id=mi.question_id,
                question=mi.question,
                question_date=mi.question_date,
            )
            scores = result["scores"]
            applied = result["names"]

        # Decide correctness:
        #   - if scoring failed, baseline correctness sticks (the agent
        #     produced no usable context regardless of transforms)
        #   - otherwise: if a gold-session turn now exceeds the
        #     mock-defined threshold AND was a real candidate, flip
        correct = mi.correct_baseline
        if not mi.score_error and applied:
            gold = set(mi.answer_session_ids)
            candidates = set(mi.candidate_turn_ids) or set(mi.selected_turn_ids) or set(scores)
            for tid, s in scores.items():
                sid = tid.split("#", 1)[0] if "#" in tid else tid
                if sid in gold and s >= mi.gold_score_threshold and tid in candidates:
                    correct = True
                    break

        return Outcome(
            question_id=mi.question_id,
            question_type=mi.question_type,
            is_abstention=mi.is_abstention,
            answer_session_ids=mi.answer_session_ids,
            correct=correct,
            judge_label="mock-1" if correct else "mock-0",
            judge_raw=None,
            hypothesis="",
            signal=self.signal,
            selected_turn_ids=mi.selected_turn_ids,
            n_seeds=mi.n_seeds,
            n_expanded=mi.n_expanded,
            truncated=mi.truncated,
            running_tokens=mi.running_tokens,
            decisions=mi.decisions,
            scores=scores,
            ranked=mi.ranked,
            applied_transforms=tuple(applied),
            run_id="mock",
            error=None,
            score_error=mi.score_error,
        )


def _from_dict(d: dict[str, Any]) -> MockInstance:
    return MockInstance(
        question_id=d["question_id"],
        question_type=d["question_type"],
        is_abstention=bool(d.get("is_abstention", False)),
        answer_session_ids=tuple(d.get("answer_session_ids", ())),
        correct_baseline=bool(d.get("correct_baseline", False)),
        scores=dict(d.get("scores", {})),
        ranked=tuple(d.get("ranked", ())),
        selected_turn_ids=tuple(d.get("selected_turn_ids", ())),
        decisions=tuple(d.get("decisions", ())),
        truncated=bool(d.get("truncated", False)),
        score_error=str(d.get("score_error", "")),
        n_seeds=int(d.get("n_seeds", 0)),
        n_expanded=int(d.get("n_expanded", 0)),
        running_tokens=int(d.get("running_tokens", 0)),
        question=str(d.get("question", "")),
        question_date=str(d.get("question_date", "")),
        gold_score_threshold=float(d.get("gold_score_threshold", float("inf"))),
        candidate_turn_ids=tuple(d.get("candidate_turn_ids", ())),
    )
