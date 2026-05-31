"""Loop event vocabulary.

The loop's phases are activegraph behaviors firing on these custom event
types. Constants live here so detectors / runners match on shared
symbols rather than scattered string literals.

Sequence per iteration:

    loop.start                       (seed; carries iteration_id)
      -> behavior_run_baseline
    baseline.recorded                (aggregate + per-question payload)
      -> behavior_diagnose
    regime.histogram                 (counts per regime + per-failure regime)
      -> behavior_hypothesize        (or pause)
    transform.drafted                (inert; name, src, target_regime, author)
      -> behavior_static_gate
    transform.static_passed          -> behavior_sandbox_gate
    transform.static_rejected        -> behavior_iterate_after_static_reject
    transform.sandbox_passed         -> behavior_eval_diff
    transform.sandbox_rejected       -> behavior_iterate_after_sandbox_reject
    transform.eval_diff              (per-type + overall deltas)
      -> behavior_promote
    transform.promoted               -> behavior_attribute -> attribution.recorded
                                        -> behavior_iterate_after_promote
    transform.discarded              -> behavior_iterate_after_discard
    loop.iterate | loop.stopped

EVERY non-promoting candidate outcome (static_rejected, sandbox_rejected,
discarded, confirm_regression) routes through `_handle_failed_attempt`:
it counts the attempt toward the regime's consecutive-failure ceiling and
then either re-drafts, rotates to the next seam-reachable regime, or stops
cleanly. Before dedicated listeners existed for the *_rejected events, the
chain died mid-iteration on a malformed/crashing candidate and the loop
exited with `loop.stopped` never emitted (the gap-4 bug).

`loop.stopped` is terminal; its payload names the wall. A completed run
ALWAYS emits exactly one (the runner has a backstop so `stopped: None` is
structurally impossible).
"""

from __future__ import annotations

LOOP_START = "loop.start"
BASELINE_RECORDED = "baseline.recorded"
REGIME_HISTOGRAM = "regime.histogram"

TRANSFORM_DRAFTED = "transform.drafted"
TRANSFORM_STATIC_PASSED = "transform.static_passed"
TRANSFORM_STATIC_REJECTED = "transform.static_rejected"
TRANSFORM_SANDBOX_PASSED = "transform.sandbox_passed"
TRANSFORM_SANDBOX_REJECTED = "transform.sandbox_rejected"
TRANSFORM_EVAL_DIFF = "transform.eval_diff"
TRANSFORM_PROMOTED = "transform.promoted"
TRANSFORM_DISCARDED = "transform.discarded"

ATTRIBUTION_RECORDED = "attribution.recorded"
LOOP_ITERATE = "loop.iterate"
LOOP_STOPPED = "loop.stopped"

ALL = (
    LOOP_START,
    BASELINE_RECORDED,
    REGIME_HISTOGRAM,
    TRANSFORM_DRAFTED,
    TRANSFORM_STATIC_PASSED,
    TRANSFORM_STATIC_REJECTED,
    TRANSFORM_SANDBOX_PASSED,
    TRANSFORM_SANDBOX_REJECTED,
    TRANSFORM_EVAL_DIFF,
    TRANSFORM_PROMOTED,
    TRANSFORM_DISCARDED,
    ATTRIBUTION_RECORDED,
    LOOP_ITERATE,
    LOOP_STOPPED,
)
