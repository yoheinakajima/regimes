"""The loop's behaviors as real @behavior registrations.

Each phase of the loop is implemented as one activegraph behavior. The
runtime drives the chain; the only Python "orchestration" outside the
runtime is the seed `loop.start` emit and the post-run pluck of the
final report. Per the prompt: "every step is an event in the real
event log."

Behaviors here are imported once; the module snapshots them so the
loop's runner can construct a `Runtime(graph, behaviors=...)` and drive
the chain WITHOUT mutating the global registry. That matters because
the agent's `retrieve()` clears+restores the global registry on every
call; if loop behaviors lived in the global registry they'd be erased
mid-iteration. We pass them explicitly to Runtime instead.

The eval backend, transform author, and instance list are passed via
the seed event's payload — but Python callables don't survive the
JSON-shaped payload contract. So we hold them in a process-level
context dict keyed by `iteration_id`. The loop runner sets the entry
before emitting `loop.start` and clears it on exit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from activegraph import behavior, get_registry

from regimes.agent import transforms as _agent_transforms
from regimes.eval.types import EvalResult
from regimes.loop import events as E
from regimes.loop import gates as _gates
from regimes.loop.attribute import attribute as _attribute
from regimes.loop.hypothesize import DraftedTransform, StubAuthor
from regimes.loop.regimes import (
    WELL_RANKED_K,
    classify,
    format_histogram,
    histogram,
    is_seam_reachable,
)


def _outcome_summary(o, *, well_ranked_k: int = WELL_RANKED_K) -> dict[str, Any]:
    """Self-justifying per-question summary for persistence.

    Carries the EVIDENCE-LEVEL signals the detectors used to assign
    the regime label — so every label in the report is auditable
    against the same numbers the detector saw. The earlier summary
    only held qid/correct/regime, which made labels unverifiable
    (e.g. an eac54add classified as budget-truncation looked the
    same on disk as a legitimate one even though the underlying
    signals disagreed).

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


# ===========================================================================
# Process-level context (one entry per active iteration).
# Carries Python objects (eval backends, author, callables, instance
# lists) that can't ride in event payloads.
# ===========================================================================


@dataclass
class LoopContext:
    iteration_id: str
    eval_backend: Any
    author: Any
    instances: list[Any]
    confirm_instances: list[Any] | None
    baseline: EvalResult | None = None
    last_result: EvalResult | None = None
    max_consecutive_discards: int = 3
    consecutive_discards: int = 0
    pause_after: str | None = None        # "histogram" | None
    # Live record of every drafted transform's outcome on OPTIMIZE.
    transform_log: list[dict[str, Any]] = None
    # The compiled callable + name of the current drafted transform
    # (gates need this; payloads carry the source string only).
    current_fn: Any = None
    current_name: str = ""
    current_target: str = ""
    halted: bool = False

    def __post_init__(self):
        if self.transform_log is None:
            self.transform_log = []


_CONTEXTS: dict[str, LoopContext] = {}
_CTX_LOCK = threading.Lock()


def set_context(ctx: LoopContext) -> None:
    with _CTX_LOCK:
        _CONTEXTS[ctx.iteration_id] = ctx


def get_context(iteration_id: str) -> LoopContext:
    with _CTX_LOCK:
        return _CONTEXTS[iteration_id]


def clear_context(iteration_id: str) -> None:
    with _CTX_LOCK:
        _CONTEXTS.pop(iteration_id, None)


# ===========================================================================
# Behaviors
# ===========================================================================


# ----- 1) Run baseline -----------------------------------------------------

@behavior(name="loop.run_baseline", on=[E.LOOP_START])
def behavior_run_baseline(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    result = lctx.eval_backend.run_on_split(lctx.instances)
    lctx.baseline = result
    lctx.last_result = result
    # Self-justifying summary: each per-question record carries the
    # evidence-level signals the detector used to assign the regime
    # label — so every label in the persisted report can be checked
    # against its basis. (Previously only qid/correct/regime were
    # persisted, which made labels unauditable.)
    outcomes_summary = [_outcome_summary(o) for o in result.outcomes]
    graph.emit(
        E.BASELINE_RECORDED,
        {
            "iteration_id": iid,
            "n": len(result.outcomes),
            "overall_accuracy": result.overall_accuracy(),
            "per_type_accuracy": result.per_type_accuracy(),
            "n_truncated": result.aggregate.get("n_truncated", 0),
            "n_errors": result.aggregate.get("n_errors", 0),
            "outcomes": outcomes_summary,
            "backend": result.backend,
            "applied_transforms": list(result.config.get("applied_transforms", ())),
        },
    )


# ----- 2) Diagnose ---------------------------------------------------------

@behavior(name="loop.diagnose", on=[E.BASELINE_RECORDED])
def behavior_diagnose(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    assert lctx.baseline is not None
    rows = histogram(lctx.baseline.outcomes)
    n_failures = sum(1 for o in lctx.baseline.outcomes if not o.correct)
    n_total = len(lctx.baseline.outcomes)
    # Per-question regime classification, for downstream attribution.
    per_qid = {
        o.question_id: ("correct" if o.correct else classify(o).name)
        for o in lctx.baseline.outcomes
    }
    graph.emit(
        E.REGIME_HISTOGRAM,
        {
            "iteration_id": iid,
            "n_total": n_total,
            "n_failures": n_failures,
            "rows": [
                {
                    "regime": r.regime,
                    "count": r.count,
                    "optimizable": r.optimizable,
                    "seam_reachable": r.seam_reachable,
                    "qids": list(r.qids),
                }
                for r in rows
            ],
            "per_question": per_qid,
            "formatted": format_histogram(rows, n_failures=n_failures, n_total=n_total),
        },
    )


# ----- 3) Hypothesize ------------------------------------------------------

@behavior(name="loop.hypothesize", on=[E.REGIME_HISTOGRAM])
def behavior_hypothesize(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)

    if lctx.pause_after == "histogram":
        # The runner caller stops the runtime via lctx.halted; we still
        # emit a marker so the event log explains the halt.
        lctx.halted = True
        graph.emit(
            E.LOOP_STOPPED,
            {
                "iteration_id": iid,
                "reason": "pause_after_histogram",
                "remaining_regimes": [
                    r["regime"] for r in event.payload["rows"] if r["count"] > 0
                ],
                "named_wall": "",
            },
        )
        return

    counts = {r["regime"]: r["count"] for r in event.payload["rows"]}
    target = _choose_target(counts, lctx)
    if not target:
        graph.emit(
            E.LOOP_STOPPED,
            {
                "iteration_id": iid,
                "reason": "no_optimizable_regime_remaining",
                "remaining_regimes": [k for k, v in counts.items() if v > 0],
                "named_wall": _name_wall(counts),
            },
        )
        return

    failing = [o for o in lctx.baseline.outcomes if not o.correct]
    drafted = lctx.author.draft(dominant_regime=target, failures=failing)
    lctx.current_target = target
    graph.emit(
        E.TRANSFORM_DRAFTED,
        {
            "iteration_id": iid,
            "name": drafted.name,
            "target_regime": drafted.target_regime,
            "author": drafted.author,
            "source": drafted.source,
            "rationale": drafted.rationale,
        },
    )


# ----- 4) Static gate ------------------------------------------------------

@behavior(name="loop.static_gate", on=[E.TRANSFORM_DRAFTED])
def behavior_static_gate(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    src = event.payload["source"]
    name = event.payload["name"]
    res = _gates.static_gate(src)
    if not res.passed:
        _log_transform(lctx, name, event.payload["target_regime"], "static_rejected",
                       reasons=list(res.reasons))
        graph.emit(
            E.TRANSFORM_STATIC_REJECTED,
            {
                "iteration_id": iid,
                "name": name,
                "reasons": list(res.reasons),
            },
        )
        return
    # Compile the source into a callable so downstream gates can run it.
    fn = _gates.compile_transform(src)
    lctx.current_fn = fn
    lctx.current_name = name
    graph.emit(
        E.TRANSFORM_STATIC_PASSED,
        {"iteration_id": iid, "name": name},
    )


# ----- 5) Sandbox gate ------------------------------------------------------

@behavior(name="loop.sandbox_gate", on=[E.TRANSFORM_STATIC_PASSED])
def behavior_sandbox_gate(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    fn = lctx.current_fn
    name = lctx.current_name
    # Pull probe scores from up to 5 outcomes — enough to exercise the
    # transform without re-running a full eval.
    probes: list[dict[str, Any]] = []
    for o in lctx.baseline.outcomes[:5]:
        probes.append({
            "scores": dict(o.scores),
            "question": "",
            "question_date": "",
        })
    res = _gates.sandbox_gate(fn, probes=probes)
    if not res.passed:
        _log_transform(lctx, name, lctx.current_target, "sandbox_rejected",
                       reasons=list(res.reasons))
        graph.emit(
            E.TRANSFORM_SANDBOX_REJECTED,
            {
                "iteration_id": iid,
                "name": name,
                "reasons": list(res.reasons),
                "n_probed": res.n_probed,
                "elapsed_s": res.elapsed_s,
            },
        )
        return
    graph.emit(
        E.TRANSFORM_SANDBOX_PASSED,
        {
            "iteration_id": iid,
            "name": name,
            "n_probed": res.n_probed,
            "elapsed_s": res.elapsed_s,
        },
    )


# ----- 6) Eval-diff gate ----------------------------------------------------

@behavior(name="loop.eval_diff", on=[E.TRANSFORM_SANDBOX_PASSED])
def behavior_eval_diff(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    fn = lctx.current_fn
    name = lctx.current_name
    target = lctx.current_target
    diff = _gates.eval_diff(
        fn=fn, fn_name=name, target_regime=target,
        baseline=lctx.baseline,
        eval_backend=lctx.eval_backend,
        instances=lctx.instances,
    )
    graph.emit(
        E.TRANSFORM_EVAL_DIFF,
        {
            "iteration_id": iid,
            "name": name,
            "target_regime": target,
            "overall_before": diff.overall_before,
            "overall_after": diff.overall_after,
            "overall_delta": diff.overall_delta,
            "per_type_delta": diff.per_type_delta,
            "regime_before": diff.regime_before,
            "regime_after": diff.regime_after,
            "target_delta": diff.target_delta,
            "transitions": [list(t) for t in diff.transitions],
        },
    )


# ----- 7) Promotion decision ------------------------------------------------

@behavior(name="loop.promote", on=[E.TRANSFORM_EVAL_DIFF])
def behavior_promote(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    name = event.payload["name"]
    target = event.payload["target_regime"]
    # Reconstruct an EvalDiff-ish shape for promotion_decision.
    diff = _gates.EvalDiff(
        overall_before=event.payload["overall_before"],
        overall_after=event.payload["overall_after"],
        overall_delta=event.payload["overall_delta"],
        per_type_delta=event.payload["per_type_delta"],
        regime_before=event.payload["regime_before"],
        regime_after=event.payload["regime_after"],
        target_regime=target,
        target_delta=event.payload["target_delta"],
        transitions=tuple(tuple(t) for t in event.payload["transitions"]),
    )
    decision = _gates.promotion_decision(diff)
    if not decision.eligible:
        _log_transform(lctx, name, target, "discarded", reasons=list(decision.reasons),
                       overall_delta=diff.overall_delta, target_delta=diff.target_delta)
        lctx.consecutive_discards += 1
        graph.emit(
            E.TRANSFORM_DISCARDED,
            {
                "iteration_id": iid,
                "name": name,
                "reasons": list(decision.reasons),
                "overall_delta": diff.overall_delta,
                "target_delta": diff.target_delta,
            },
        )
        return

    # Promote: register on the agent's seam and run a one-shot CONFIRM
    # check (if a CONFIRM set was supplied).
    _agent_transforms.promote(name, lctx.current_fn)
    lctx.consecutive_discards = 0
    confirm_delta = None
    if lctx.confirm_instances:
        # Compare CONFIRM accuracy with vs without the transform.
        # We promoted above; for the "without" baseline reading we
        # revert temporarily.
        confirm_after = lctx.eval_backend.run_on_split(lctx.confirm_instances)
        _agent_transforms.revert(name)
        confirm_before = lctx.eval_backend.run_on_split(lctx.confirm_instances)
        _agent_transforms.promote(name, lctx.current_fn)
        confirm_delta = (
            confirm_after.overall_accuracy() - confirm_before.overall_accuracy()
        )
    _log_transform(lctx, name, target, "promoted",
                   overall_delta=diff.overall_delta, target_delta=diff.target_delta,
                   confirm_delta=confirm_delta)
    graph.emit(
        E.TRANSFORM_PROMOTED,
        {
            "iteration_id": iid,
            "name": name,
            "target_regime": target,
            "overall_delta": diff.overall_delta,
            "target_delta": diff.target_delta,
            "confirm_delta": confirm_delta,
        },
    )


# ----- 8) Attribution -------------------------------------------------------

@behavior(name="loop.attribute", on=[E.TRANSFORM_PROMOTED])
def behavior_attribute(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    name = lctx.current_name
    # Re-run with the (just-promoted) transform so we can structurally
    # compare to the original baseline. Promotion has already installed
    # the transform on the agent's seam, so a plain run_on_split picks
    # it up.
    after = lctx.eval_backend.run_on_split(lctx.instances)
    lctx.last_result = after
    att = _attribute(lctx.baseline, after)
    graph.emit(
        E.ATTRIBUTION_RECORDED,
        {
            "iteration_id": iid,
            "name": name,
            "transitions": [list(t) for t in att.transitions],
            "n_recovered": att.n_recovered,
            "n_introduced": att.n_introduced,
        },
    )


# ----- 9) Stop / iterate ----------------------------------------------------

@behavior(name="loop.iterate_decision_on_promote", on=[E.ATTRIBUTION_RECORDED])
def behavior_iterate_after_promote(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    _emit_next_step(graph, lctx, iid)


@behavior(name="loop.iterate_decision_on_discard", on=[E.TRANSFORM_DISCARDED])
def behavior_iterate_after_discard(event, graph, ctx) -> None:  # noqa: ARG001
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    if lctx.consecutive_discards >= lctx.max_consecutive_discards:
        rows = histogram(lctx.last_result.outcomes if lctx.last_result else
                          lctx.baseline.outcomes)
        counts = {r.regime: r.count for r in rows}
        graph.emit(
            E.LOOP_STOPPED,
            {
                "iteration_id": iid,
                "reason": "max_consecutive_discards",
                "remaining_regimes": [k for k, v in counts.items() if v > 0],
                "named_wall": _name_wall(counts),
            },
        )
        return
    _emit_next_step(graph, lctx, iid)


# ===========================================================================
# Helpers
# ===========================================================================


def _emit_next_step(graph, lctx: LoopContext, iid: str) -> None:
    """Decide whether to iterate (re-fire diagnose) or stop with a named
    wall. Reads the latest result's failures and applies the
    optimizable-remaining check."""
    latest = lctx.last_result or lctx.baseline
    if latest is None:
        return
    rows = histogram(latest.outcomes)
    counts = {r.regime: r.count for r in rows}
    if not _any_optimizable_remaining(counts):
        graph.emit(
            E.LOOP_STOPPED,
            {
                "iteration_id": iid,
                "reason": "no_optimizable_regime_remaining",
                "remaining_regimes": [k for k, v in counts.items() if v > 0],
                "named_wall": _name_wall(counts),
            },
        )
        return
    # Iterate: re-emit BASELINE_RECORDED-style content so diagnose fires
    # again on the latest result. We do this by emitting LOOP_ITERATE
    # which a tiny re-baseline behavior listens for.
    graph.emit(
        E.LOOP_ITERATE,
        {"iteration_id": iid, "round": lctx.consecutive_discards},
    )


@behavior(name="loop.rebaseline", on=[E.LOOP_ITERATE])
def behavior_rebaseline(event, graph, ctx) -> None:  # noqa: ARG001
    """On LOOP_ITERATE, re-fire the diagnose step against `last_result`.

    We treat `last_result` (possibly post-transform) as the new
    baseline for THIS round's diagnose. The original `baseline` stays
    pinned for attribution."""
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    base_for_round = lctx.last_result or lctx.baseline
    # Update the "baseline used by diagnose" via the existing slot; the
    # original baseline is preserved on lctx for attribution.
    lctx.baseline = base_for_round
    outcomes_summary = [_outcome_summary(o) for o in base_for_round.outcomes]
    graph.emit(
        E.BASELINE_RECORDED,
        {
            "iteration_id": iid,
            "n": len(base_for_round.outcomes),
            "overall_accuracy": base_for_round.overall_accuracy(),
            "per_type_accuracy": base_for_round.per_type_accuracy(),
            "n_truncated": base_for_round.aggregate.get("n_truncated", 0),
            "n_errors": base_for_round.aggregate.get("n_errors", 0),
            "outcomes": outcomes_summary,
            "backend": base_for_round.backend,
            "applied_transforms": list(
                base_for_round.config.get("applied_transforms", ())
            ),
            "round_marker": True,
        },
    )


def _any_optimizable_remaining(counts: dict[str, int]) -> bool:
    """A regime is "remaining and optimizable" iff its count > 0 and the
    taxonomy marks it as seam-reachable + optimizable."""
    from regimes.loop.regimes import REGIMES as _all_regimes  # local: dynamic
    reg = _all_regimes()
    for name, c in counts.items():
        if c <= 0:
            continue
        r = reg.get(name)
        if r is None:
            continue
        if r.optimizable and r.seam_reachable:
            return True
    return False


def _name_wall(counts: dict[str, int]) -> str:
    """Construct the named-wall string for the loop.stopped payload.

    Lists the remaining unreachable regimes and what would be needed to
    address each. Pure description; no recommendation about which to
    pursue."""
    from regimes.loop.regimes import REGIMES as _all_regimes  # local
    reg = _all_regimes()
    fragments: list[str] = []
    for name, c in sorted(counts.items()):
        if c <= 0:
            continue
        r = reg.get(name)
        if r is None or (r.optimizable and r.seam_reachable):
            continue
        if name == "retrieval-signal-gap":
            fix = "signal change (better embedder / scorer)"
        elif name == "assemble-internal":
            fix = "assemble() internals change (reader prompt / context format)"
        elif name == "scoring-error":
            fix = "fix the scoring-step exception (e.g. input truncation before embedding)"
        else:
            fix = "outside the score-transform action space"
        fragments.append(f"{name}={c} → {fix}")
    return "; ".join(fragments) if fragments else "no remaining failures"


def _choose_target(counts: dict[str, int], lctx: LoopContext) -> str:
    """Pick the highest-count optimizable+seam-reachable regime."""
    from regimes.loop.regimes import REGIMES as _all_regimes
    reg = _all_regimes()
    candidates: list[tuple[int, str]] = []
    for name, c in counts.items():
        if c <= 0:
            continue
        r = reg.get(name)
        if r is None or not (r.optimizable and r.seam_reachable):
            continue
        candidates.append((-c, name))
    candidates.sort()
    return candidates[0][1] if candidates else ""


def _log_transform(
    lctx: LoopContext,
    name: str,
    target: str,
    status: str,
    **fields: Any,
) -> None:
    """Best-of-N audit: every drafted transform attempt is recorded.

    Held-out discipline: the loop never promotes on the strength of a
    single number; the transform log lets a reviewer reconstruct the
    full search and report the headline as best-of-N."""
    rec: dict[str, Any] = {
        "name": name,
        "target_regime": target,
        "status": status,
    }
    rec.update(fields)
    lctx.transform_log.append(rec)


# ===========================================================================
# Snapshot at import time so the runner can pass behaviors=... explicitly
# to Runtime — keeps the loop's behaviors out of the global registry that
# `agent.retrieve()` clears on every call.
# ===========================================================================

LOOP_BEHAVIORS = [b for b in get_registry() if b.name.startswith("loop.")]
