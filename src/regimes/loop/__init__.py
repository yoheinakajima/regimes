"""The regimes loop.

A retrieval-improvement loop that runs natively on the ActiveGraph
runtime. Every phase — `run.baseline`, `diagnose`, `transform.drafted`,
gate results, `transform.promoted` / `transform.discarded`, `attribute`,
`loop.stopped` — is a real activegraph event in the loop's event log.

Public surface:

    from regimes.loop import run_loop, MockEval
    g = run_loop(eval_backend=MockEval(...), split=split, instances=insts,
                 pause_after="histogram")
    # `g.events` is the audit log; the histogram has been printed.

The loop's event chain (one full iteration):

    loop.start
      -> behavior_run_baseline   -> baseline.recorded
      -> behavior_diagnose       -> regime.histogram   [PAUSE POINT]
      -> behavior_hypothesize    -> transform.drafted
      -> behavior_static_gate    -> transform.static_{passed,rejected}
      -> behavior_sandbox_gate   -> transform.sandbox_{passed,rejected}
      -> behavior_eval_diff      -> transform.eval_diff
      -> behavior_promote        -> transform.{promoted,discarded}
      -> behavior_attribute      -> attribution.recorded   (on promotion)
      -> behavior_iterate        -> loop.iterate | loop.stopped

`loop.iterate` re-fires diagnose for the next round; `loop.stopped` is
terminal and its payload names the wall (which regimes remain, why the
score-transform action space can't reach them).
"""

from __future__ import annotations

from regimes.loop.events import (
    ATTRIBUTION_RECORDED,
    BASELINE_RECORDED,
    LOOP_ITERATE,
    LOOP_START,
    LOOP_STOPPED,
    REGIME_HISTOGRAM,
    TRANSFORM_DISCARDED,
    TRANSFORM_DRAFTED,
    TRANSFORM_EVAL_DIFF,
    TRANSFORM_PROMOTED,
    TRANSFORM_SANDBOX_PASSED,
    TRANSFORM_SANDBOX_REJECTED,
    TRANSFORM_STATIC_PASSED,
    TRANSFORM_STATIC_REJECTED,
)
from regimes.loop.mock_eval import MockEval, MockInstance
from regimes.loop.regimes import (
    REGIMES,
    Regime,
    classify,
    histogram,
    is_seam_reachable,
)
from regimes.loop.runner import LoopReport, run_loop

__all__ = [
    "ATTRIBUTION_RECORDED",
    "BASELINE_RECORDED",
    "LOOP_ITERATE",
    "LOOP_START",
    "LOOP_STOPPED",
    "REGIMES",
    "REGIME_HISTOGRAM",
    "TRANSFORM_DISCARDED",
    "TRANSFORM_DRAFTED",
    "TRANSFORM_EVAL_DIFF",
    "TRANSFORM_PROMOTED",
    "TRANSFORM_SANDBOX_PASSED",
    "TRANSFORM_SANDBOX_REJECTED",
    "TRANSFORM_STATIC_PASSED",
    "TRANSFORM_STATIC_REJECTED",
    "LoopReport",
    "MockEval",
    "MockInstance",
    "Regime",
    "classify",
    "histogram",
    "is_seam_reachable",
    "run_loop",
]
