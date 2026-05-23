"""Top-level orchestrator: `run_loop(...)`.

Constructs the loop's Graph + Runtime, seeds `loop.start`, drains the
runtime, returns a LoopReport that summarizes what happened (carrying
the full event log for audit).

Single Python responsibility outside the runtime:
  - seed the chain with `loop.start`
  - drain the runtime
  - assemble the LoopReport from the terminal events

Everything else — baseline, diagnose, hypothesize, gates, attribute,
stop — runs as behaviors emitting events. The audit trail IS the event
log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from activegraph import Event, FrozenClock, Graph, IDGen, Runtime

from regimes.loop import events as E
from regimes.loop.behaviors import (
    LOOP_BEHAVIORS,
    LoopContext,
    clear_context,
    set_context,
)
from regimes.loop.hypothesize import DraftedTransform, StubAuthor


@dataclass
class LoopReport:
    """The loop's result. The event log is the source of truth; the
    convenience fields below are pre-extracted from terminal events."""

    iteration_id: str
    events: list[Event]
    histogram: dict[str, Any] | None      # the regime.histogram event payload
    baseline: dict[str, Any] | None       # the baseline.recorded payload
    stopped: dict[str, Any] | None        # the loop.stopped payload
    promotions: list[dict[str, Any]]      # transform.promoted payloads
    discards: list[dict[str, Any]]        # transform.discarded payloads
    attributions: list[dict[str, Any]]    # attribution.recorded payloads
    transform_log: list[dict[str, Any]] = field(default_factory=list)


def run_loop(
    *,
    eval_backend: Any,
    instances: list[Any],
    confirm_instances: list[Any] | None = None,
    author: Any = None,
    pause_after: str | None = None,
    iteration_id: str = "loop-001",
    frozen_t: str = "2026-01-01T00:00:00Z",
    max_consecutive_discards: int = 3,
) -> LoopReport:
    """Run the loop end-to-end and return a LoopReport.

    `pause_after`:
        - None        → run all phases through stop.
        - "histogram" → emit `regime.histogram` then `loop.stopped` with
                        reason="pause_after_histogram". This is the
                        live pause-discipline checkpoint: the histogram
                        is the go/no-go for spending eval budget on
                        transforms.
    """
    author = author if author is not None else StubAuthor()
    lctx = LoopContext(
        iteration_id=iteration_id,
        eval_backend=eval_backend,
        author=author,
        instances=list(instances),
        confirm_instances=list(confirm_instances) if confirm_instances else None,
        max_consecutive_discards=max_consecutive_discards,
        pause_after=pause_after,
    )
    set_context(lctx)

    try:
        graph = Graph(
            ids=IDGen(),
            clock=FrozenClock(frozen_t),
            run_id=f"regimes-loop-{iteration_id}",
        )
        # Pass loop behaviors EXPLICITLY — bypasses the global registry
        # that agent.retrieve() clears on every call.
        rt = Runtime(graph, behaviors=LOOP_BEHAVIORS)

        seed = Event(
            id=graph.ids.event(),
            type=E.LOOP_START,
            payload={"iteration_id": iteration_id},
            actor="caller",
            caused_by=None,
            timestamp=graph.clock.now(),
        )
        graph.emit(seed)
        rt.run_until_idle()

        # Pluck terminal payloads for the report.
        histogram_payload = None
        baseline_payload = None
        stopped_payload = None
        promotions: list[dict[str, Any]] = []
        discards: list[dict[str, Any]] = []
        attributions: list[dict[str, Any]] = []
        for ev in graph.events:
            if ev.type == E.REGIME_HISTOGRAM and histogram_payload is None:
                histogram_payload = dict(ev.payload)
            elif ev.type == E.BASELINE_RECORDED and baseline_payload is None:
                baseline_payload = dict(ev.payload)
            elif ev.type == E.LOOP_STOPPED:
                stopped_payload = dict(ev.payload)
            elif ev.type == E.TRANSFORM_PROMOTED:
                promotions.append(dict(ev.payload))
            elif ev.type == E.TRANSFORM_DISCARDED:
                discards.append(dict(ev.payload))
            elif ev.type == E.ATTRIBUTION_RECORDED:
                attributions.append(dict(ev.payload))

        return LoopReport(
            iteration_id=iteration_id,
            events=list(graph.events),
            histogram=histogram_payload,
            baseline=baseline_payload,
            stopped=stopped_payload,
            promotions=promotions,
            discards=discards,
            attributions=attributions,
            transform_log=list(lctx.transform_log),
        )
    finally:
        clear_context(iteration_id)
