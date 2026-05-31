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

Target-agnostic phase 1: each behavior reaches the LongMemEval-specific
bits (outcome_summary, name_wall, the gate functions, the install/
revert seam) via `lctx.target.*` rather than direct module imports. The
default target is `LongMemEvalTarget`, built by the runner from the
caller's `eval_backend` + `author` — so existing callers see no change
beyond the indirection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from activegraph import behavior, get_registry

from regimes.eval.types import EvalResult
from regimes.loop import events as E
from regimes.loop.attribute import attribute as _attribute
from regimes.loop.gates import EvalDiff
from regimes.target import Target


# Backward-compat re-export: tests import `_outcome_summary` from this
# module (the function moved to regimes.targets.longmemeval). Keeping
# the symbol importable here avoids breaking external callers.
from regimes.targets.longmemeval.outcome_summary import outcome_summary as _outcome_summary  # noqa: E402,F401


# ===========================================================================
# Process-level context (one entry per active iteration).
# Carries Python objects (eval backends, author, callables, instance
# lists) that can't ride in event payloads.
# ===========================================================================


@dataclass
class LoopContext:
    iteration_id: str
    target: Target
    instances: list[Any]
    confirm_instances: list[Any] | None
    baseline: EvalResult | None = None
    last_result: EvalResult | None = None
    # The two CONFIRM (held-out) eval snapshots taken at the promotion
    # gate: baseline-on-CONFIRM (transform reverted) and
    # transform-on-CONFIRM. Stashed by `behavior_promote` so
    # `behavior_attribute` can compute held-out transitions, and so the
    # promotion event can persist the per-question held-out outcomes.
    confirm_baseline: EvalResult | None = None
    confirm_transform: EvalResult | None = None
    max_consecutive_discards: int = 3
    consecutive_discards: int = 0
    # Regimes that have hit max_consecutive_discards and been retired from
    # drafting THIS run. The loop rotates to the next un-attempted
    # optimizable+seam-reachable regime instead of stopping the whole loop
    # the moment one regime stalls — so with budget-truncation AND
    # assemble-internal both reachable, both get attempted before stop.
    exhausted_regimes: set[str] = None
    # Generous global guard so a pathological promote/discard cycle can't
    # spin forever. Sized to comfortably attempt every regime several times.
    max_iterations: int = 200
    iterations: int = 0
    pause_after: str | None = None        # "histogram" | None
    # Live record of every drafted transform's outcome on OPTIMIZE.
    transform_log: list[dict[str, Any]] = None
    # The compiled callable + name of the current drafted transform
    # (gates need this; payloads carry the source string only).
    current_fn: Any = None
    current_name: str = ""
    current_target: str = ""
    # The source string of the currently-drafted transform. Carried on
    # ctx so each transform_log entry can record the actual code the
    # author produced (the audit trail otherwise loses authored code
    # past the TRANSFORM_DRAFTED event).
    current_source: str = ""
    current_author: str = ""
    current_transform_type: str = ""
    halted: bool = False

    def __post_init__(self):
        if self.transform_log is None:
            self.transform_log = []
        if self.exhausted_regimes is None:
            self.exhausted_regimes = set()

    # ---- back-compat accessors --------------------------------------------
    # Pre-Target-interface callers expected `lctx.eval_backend`. The
    # backend now lives on the target; expose it as a property so the
    # external surface is unchanged.

    @property
    def eval_backend(self) -> Any:
        return self.target.eval_backend


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
    result = lctx.target.eval_backend.run_on_split(lctx.instances)
    lctx.baseline = result
    lctx.last_result = result
    # Self-justifying summary: each per-question record carries the
    # evidence-level signals the detector used to assign the regime
    # label — so every label in the persisted report can be checked
    # against its basis. (Previously only qid/correct/regime were
    # persisted, which made labels unauditable.)
    outcomes_summary = [lctx.target.outcome_summary(o) for o in result.outcomes]
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
    tax = lctx.target.taxonomy
    rows = tax.histogram(lctx.baseline.outcomes)
    n_failures = sum(1 for o in lctx.baseline.outcomes if not o.correct)
    n_total = len(lctx.baseline.outcomes)
    # Per-question regime classification, for downstream attribution.
    per_qid = {
        o.question_id: ("correct" if o.correct else tax.classify(o).name)
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
            "formatted": tax.format_histogram(rows, n_failures=n_failures, n_total=n_total),
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
    chosen = _choose_target(counts, lctx)
    if not chosen:
        graph.emit(
            E.LOOP_STOPPED,
            {
                "iteration_id": iid,
                "reason": "no_optimizable_regime_remaining",
                "remaining_regimes": [k for k, v in counts.items() if v > 0],
                "named_wall": lctx.target.taxonomy.name_wall(counts),
            },
        )
        return

    failing = [o for o in lctx.baseline.outcomes if not o.correct]
    drafted = lctx.target.action_space.draft(
        dominant_regime=chosen, failures=failing,
    )
    lctx.current_target = chosen
    lctx.current_source = drafted.source
    lctx.current_author = drafted.author
    lctx.current_transform_type = drafted.transform_type
    graph.emit(
        E.TRANSFORM_DRAFTED,
        {
            "iteration_id": iid,
            "name": drafted.name,
            "target_regime": drafted.target_regime,
            "transform_type": drafted.transform_type,
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
    aspace = lctx.target.action_space
    res = aspace.static_gate(src)
    if not res.passed:
        _log_transform(lctx, name, event.payload["target_regime"], "static_rejected",
                       reasons=list(res.reasons), source=src,
                       author=lctx.current_author)
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
    fn = aspace.compile(src)
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
    aspace = lctx.target.action_space
    # The probe shape is per-target (score-transform inputs for
    # LongMemEval; a different shape for SQL in Phase 2). Built by the
    # ActionSpace from the baseline outcomes.
    probes = aspace.build_probes(lctx.baseline)
    res = aspace.sandbox_gate(fn, probes=probes)
    if not res.passed:
        _log_transform(lctx, name, lctx.current_target, "sandbox_rejected",
                       reasons=list(res.reasons), source=lctx.current_source,
                       author=lctx.current_author)
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
    target_regime = lctx.current_target
    diff = lctx.target.action_space.eval_diff(
        fn=fn, fn_name=name, target_regime=target_regime,
        baseline=lctx.baseline,
        eval_backend=lctx.target.eval_backend,
        instances=lctx.instances,
    )
    graph.emit(
        E.TRANSFORM_EVAL_DIFF,
        {
            "iteration_id": iid,
            "name": name,
            "target_regime": target_regime,
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
    target_regime = event.payload["target_regime"]
    aspace = lctx.target.action_space
    # Reconstruct an EvalDiff-ish shape for promotion_decision.
    diff = EvalDiff(
        overall_before=event.payload["overall_before"],
        overall_after=event.payload["overall_after"],
        overall_delta=event.payload["overall_delta"],
        per_type_delta=event.payload["per_type_delta"],
        regime_before=event.payload["regime_before"],
        regime_after=event.payload["regime_after"],
        target_regime=target_regime,
        target_delta=event.payload["target_delta"],
        transitions=tuple(tuple(t) for t in event.payload["transitions"]),
    )
    decision = aspace.promotion_decision(diff)
    if not decision.eligible:
        _log_transform(lctx, name, target_regime, "discarded",
                       reasons=list(decision.reasons),
                       overall_delta=diff.overall_delta, target_delta=diff.target_delta,
                       source=lctx.current_source, author=lctx.current_author)
        # The consecutive-failure counter is bumped once, centrally, in
        # `_handle_failed_attempt` (fired off TRANSFORM_DISCARDED) so that
        # discards, confirm-regressions, static-rejects and sandbox-rejects
        # all count identically toward the regime's ceiling.
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

    # Promote: install on the target's action-space seam and run a
    # one-shot CONFIRM check (if a CONFIRM set was supplied).
    aspace.install(name, lctx.current_fn)
    confirm_delta = None
    confirm_before = None
    confirm_after = None
    if lctx.confirm_instances:
        # Compare CONFIRM accuracy with vs without the transform.
        # We promoted above; for the "without" baseline reading we
        # revert temporarily.
        backend = lctx.target.eval_backend
        confirm_after = backend.run_on_split(lctx.confirm_instances)
        aspace.revert(name)
        confirm_before = backend.run_on_split(lctx.confirm_instances)
        aspace.install(name, lctx.current_fn)
        # Pin both held-out snapshots so the promotion event can persist
        # the per-question CONFIRM outcomes and `behavior_attribute` can
        # compute held-out transitions from the same two runs.
        lctx.confirm_baseline = confirm_before
        lctx.confirm_transform = confirm_after
        confirm_delta = (
            confirm_after.overall_accuracy() - confirm_before.overall_accuracy()
        )
        # Confirm gate: held-out delta must clear the threshold.
        # Default 0.0 = "must not regress". Reader non-determinism
        # is ~+-0.02-0.04 on LongMemEval; a threshold of +0.0 still
        # admits within-noise promotions — callers may set higher
        # (e.g. +0.02) to require clearing the noise band.
        confirm_threshold = getattr(aspace, "confirm_threshold", 0.0)
        if confirm_delta < confirm_threshold:
            aspace.revert(name)
            _log_transform(lctx, name, target_regime, "discarded",
                           reasons=["confirm_regression"],
                           overall_delta=diff.overall_delta,
                           target_delta=diff.target_delta,
                           confirm_delta=confirm_delta,
                           confirm_threshold=confirm_threshold,
                           source=lctx.current_source,
                           author=lctx.current_author)
            # Counter bump happens centrally in _handle_failed_attempt.
            graph.emit(
                E.TRANSFORM_DISCARDED,
                {
                    "iteration_id": iid,
                    "name": name,
                    "reasons": ["confirm_regression"],
                    "overall_delta": diff.overall_delta,
                    "target_delta": diff.target_delta,
                    "confirm_delta": confirm_delta,
                    "confirm_threshold": confirm_threshold,
                },
            )
            return
    lctx.consecutive_discards = 0
    _log_transform(lctx, name, target_regime, "promoted",
                   overall_delta=diff.overall_delta, target_delta=diff.target_delta,
                   confirm_delta=confirm_delta, source=lctx.current_source,
                   author=lctx.current_author)
    # Persist the per-question held-out (CONFIRM) outcomes for BOTH the
    # baseline-on-CONFIRM and transform-on-CONFIRM runs, in the same
    # shape as baseline.outcomes (question_id, question_type, correct,
    # regime, is_abstention, ...). Only the aggregate confirm_delta was
    # persisted before, which made per-type held-out deltas, held-out
    # flip tables and abstention movement impossible to reconstruct from
    # the saved report. Empty lists when no CONFIRM set was supplied.
    confirm_baseline_outcomes = (
        [lctx.target.outcome_summary(o) for o in confirm_before.outcomes]
        if confirm_before is not None else []
    )
    confirm_transform_outcomes = (
        [lctx.target.outcome_summary(o) for o in confirm_after.outcomes]
        if confirm_after is not None else []
    )
    graph.emit(
        E.TRANSFORM_PROMOTED,
        {
            "iteration_id": iid,
            "name": name,
            "target_regime": target_regime,
            "overall_delta": diff.overall_delta,
            "target_delta": diff.target_delta,
            "confirm_delta": confirm_delta,
            "confirm_baseline_outcomes": confirm_baseline_outcomes,
            "confirm_transform_outcomes": confirm_transform_outcomes,
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
    after = lctx.target.eval_backend.run_on_split(lctx.instances)
    lctx.last_result = after
    att = _attribute(lctx.baseline, after, taxonomy=lctx.target.taxonomy)
    payload = {
        "iteration_id": iid,
        "name": name,
        "split": "optimize",
        # Bare (qid, from, to) triples — kept for backward compatibility.
        "transitions": [list(t) for t in att.transitions],
        # Direction made explicit: BOTH wrong→right ("gained") and
        # right→wrong ("lost") flips, plus wrong→wrong regime shifts.
        # Required to prove (or disprove) per-category regressions.
        "transition_rows": list(att.directed_rows()),
        "n_recovered": att.n_recovered,
        "n_introduced": att.n_introduced,
    }
    # Held-out (CONFIRM) attribution from the same two snapshots the
    # promotion gate took: baseline-on-CONFIRM vs transform-on-CONFIRM.
    # Carries the same bidirectional, direction-explicit shape so
    # held-out regressions are visible, not just OPTIMIZE ones.
    if lctx.confirm_baseline is not None and lctx.confirm_transform is not None:
        catt = _attribute(
            lctx.confirm_baseline, lctx.confirm_transform,
            taxonomy=lctx.target.taxonomy,
        )
        payload["confirm_transitions"] = [list(t) for t in catt.transitions]
        payload["confirm_transition_rows"] = list(catt.directed_rows())
        payload["confirm_n_recovered"] = catt.n_recovered
        payload["confirm_n_introduced"] = catt.n_introduced
    graph.emit(E.ATTRIBUTION_RECORDED, payload)


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
    _handle_failed_attempt(graph, lctx, iid)


@behavior(name="loop.iterate_decision_on_static_reject", on=[E.TRANSFORM_STATIC_REJECTED])
def behavior_iterate_after_static_reject(event, graph, ctx) -> None:  # noqa: ARG001
    """A malformed/invalid candidate (e.g. unclosed brace) is a FAILED
    attempt for the regime — exactly like a discard. Before this behavior
    existed, TRANSFORM_STATIC_REJECTED had no listener: the chain died
    mid-iteration, the counter never advanced, the regime never exhausted,
    and the loop exited with `stopped: None` (gap 4). Routing it through
    the same handler makes static-rejects count toward the ceiling and
    trigger rotation/clean-stop."""
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    _handle_failed_attempt(graph, lctx, iid)


@behavior(name="loop.iterate_decision_on_sandbox_reject", on=[E.TRANSFORM_SANDBOX_REJECTED])
def behavior_iterate_after_sandbox_reject(event, graph, ctx) -> None:  # noqa: ARG001
    """A candidate that crashes (or misbehaves) in the sandbox is a FAILED
    attempt for the regime. Same dead-end as static-reject before this
    listener existed — now counted + rotated like a discard."""
    iid = event.payload["iteration_id"]
    lctx = get_context(iid)
    _handle_failed_attempt(graph, lctx, iid)


# ===========================================================================
# Helpers
# ===========================================================================


def _handle_failed_attempt(graph, lctx: LoopContext, iid: str) -> None:
    """Single sink for EVERY non-promoting candidate outcome.

    A candidate can fail as: discarded (OPTIMIZE gates), confirm_regression
    (held-out), static_rejected (malformed code), or sandbox_rejected
    (crashes in sandbox). All four are FAILED ATTEMPTS for the current
    regime and must count identically:

      1. bump the regime's consecutive-failure counter,
      2. if it hit the ceiling, RETIRE the regime (add to
         `exhausted_regimes`) and reset the counter so the next regime
         starts fresh,
      3. hand off to `_emit_next_step`, which decides draft-again /
         rotate-to-next-regime / stop-cleanly — and ALWAYS emits a
         terminal `loop.stopped` when nothing reachable remains.

    Centralizing the bump here (rather than scattering `+= 1` across the
    promote behavior) is what makes a regime whose author only produces
    garbage exhaust and rotate instead of looping forever or falling out
    of the chain with `stopped: None`."""
    lctx.consecutive_discards += 1
    if lctx.consecutive_discards >= lctx.max_consecutive_discards:
        # This regime is exhausted: drafting against it kept failing
        # (discard / regression / malformed / crashing — any mix). Retire
        # it and let _emit_next_step ROTATE to the next un-attempted
        # optimizable+seam-reachable regime instead of stopping the whole
        # loop. The loop only stops once EVERY reachable regime is
        # exhausted (handled in _emit_next_step via _choose_target).
        lctx.exhausted_regimes.add(lctx.current_target)
        lctx.consecutive_discards = 0
    _emit_next_step(graph, lctx, iid)


def _emit_next_step(graph, lctx: LoopContext, iid: str) -> None:
    """Decide whether to iterate (re-fire diagnose) or stop with a named
    wall.

    Stops iff there is no un-exhausted optimizable+seam-reachable regime
    left to attempt (`_choose_target` returns "" once every reachable
    regime is either gone from the histogram or retired into
    `exhausted_regimes`) — OR the generous global iteration guard trips."""
    latest = lctx.last_result or lctx.baseline
    if latest is None:
        return
    rows = lctx.target.taxonomy.histogram(latest.outcomes)
    counts = {r.regime: r.count for r in rows}
    if not _choose_target(counts, lctx):
        graph.emit(
            E.LOOP_STOPPED,
            {
                "iteration_id": iid,
                "reason": "no_optimizable_regime_remaining",
                "remaining_regimes": [k for k, v in counts.items() if v > 0],
                "named_wall": lctx.target.taxonomy.name_wall(counts),
            },
        )
        return
    if lctx.iterations >= lctx.max_iterations:
        graph.emit(
            E.LOOP_STOPPED,
            {
                "iteration_id": iid,
                "reason": "max_iterations",
                "remaining_regimes": [k for k, v in counts.items() if v > 0],
                "named_wall": lctx.target.taxonomy.name_wall(counts),
            },
        )
        return
    lctx.iterations += 1
    # Iterate: re-emit BASELINE_RECORDED-style content so diagnose fires
    # again on the latest result. We do this by emitting LOOP_ITERATE
    # which a tiny re-baseline behavior listens for.
    graph.emit(
        E.LOOP_ITERATE,
        {"iteration_id": iid, "round": lctx.iterations},
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
    outcomes_summary = [lctx.target.outcome_summary(o) for o in base_for_round.outcomes]
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


def _choose_target(counts: dict[str, int], lctx: LoopContext) -> str:
    """Pick the highest-count optimizable+seam-reachable regime that has
    NOT yet been exhausted this run.

    Regimes retired into `lctx.exhausted_regimes` (each hit
    max_consecutive_discards) are skipped so the loop rotates through every
    reachable regime before giving up. Returns "" when nothing reachable
    remains — the loop's stop condition."""
    reg = lctx.target.taxonomy.REGIMES()
    candidates: list[tuple[int, str]] = []
    for name, c in counts.items():
        if c <= 0:
            continue
        if name in lctx.exhausted_regimes:
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
        # The action-space seam this candidate used (score-transform /
        # assembly-transform / reader-prompt-transform). Recorded on EVERY
        # entry so the audit reveals which seam each candidate exercised —
        # required to verify reader-prompt-transforms actually fire.
        "transform_type": lctx.current_transform_type,
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
