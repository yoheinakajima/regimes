"""LongMemEval per-Outcome audit projection.

Moved out of `regimes.loop.behaviors._outcome_summary`. The loop now
calls `target.outcome_summary(o)` and `LongMemEvalTarget.outcome_summary`
delegates here. A backward-compat re-export in `regimes.loop.behaviors`
keeps existing imports (`from regimes.loop.behaviors import _outcome_summary`)
working.

Self-justifying per-question summary for persistence: carries the
EVIDENCE-LEVEL signals the detectors used to assign the regime label —
so every label in the report is auditable against the same numbers the
detector saw."""

from __future__ import annotations

from typing import Any

from regimes.eval.types import Outcome
from regimes.loop.regimes import WELL_RANKED_K, classify


def outcome_summary(o: Outcome, *, well_ranked_k: int = WELL_RANKED_K) -> dict[str, Any]:
    """Build the per-question summary the loop emits on baseline.recorded.

    Fields:
      gold_evidence_turn_ids       — the evidence turns from the
                                     dataset's per-turn markers
      evidence_rank_positions      — {turn_id -> 0-indexed rank in
                                     `ranked`}; missing entries mean
                                     the evidence didn't appear in
                                     the ranking at all
      evidence_in_scores           — at least one evidence turn was
                                     scored
      evidence_max_score           — best score on any evidence turn
      evidence_well_ranked         — evidence turns in top-K
      evidence_selected            — evidence turns that survived
                                     into selected_turn_ids
      evidence_dropped_at_budget   — evidence turns in decisions with
                                     included=False, reason='budget'
      evidence_coverage            — fraction of well-ranked evidence
                                     in selected, the key signal for
                                     the crowding vs assemble-internal
                                     split; None when no well-ranked
                                     evidence exists
    """
    regime_name = classify(o).name if not o.correct else "correct"
    evidence_ranks = o.evidence_rank_positions() if o.has_evidence_turn_ids() else {}
    well_ranked = list(o.evidence_ranked_top_k(well_ranked_k)) \
        if o.has_evidence_turn_ids() else []
    selected_evidence = list(o.evidence_selected()) \
        if o.has_evidence_turn_ids() else []
    dropped_evidence = list(o.evidence_dropped_at_budget()) \
        if o.has_evidence_turn_ids() else []
    coverage: float | None = None
    if well_ranked:
        n_in_sel = sum(1 for t in well_ranked if t in o.selected_turn_ids)
        coverage = n_in_sel / len(well_ranked)
    return {
        "question_id": o.question_id,
        "question_type": o.question_type,
        "correct": o.correct,
        "regime": regime_name,
        "is_abstention": o.is_abstention,
        "truncated": o.truncated,
        "n_selected": len(o.selected_turn_ids),
        "score_error": bool(o.score_error),
        # ---- evidence-level signals (the detector's actual inputs) ----
        "gold_evidence_turn_ids": list(o.gold_evidence_turn_ids),
        "evidence_rank_positions": evidence_ranks,
        "evidence_in_scores": o.evidence_in_scores() if o.has_evidence_turn_ids() else False,
        "evidence_max_score": o.evidence_max_score() if o.has_evidence_turn_ids() else 0.0,
        "evidence_well_ranked": well_ranked,
        "evidence_selected": selected_evidence,
        "evidence_dropped_at_budget": dropped_evidence,
        "evidence_coverage": coverage,
        "well_ranked_k": well_ranked_k,
    }
