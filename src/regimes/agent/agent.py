"""Top-level agent: ingest + retrieve.

This is the only place that constructs a Graph + Runtime. Behaviors are
imported (which side-effect-registers them in activegraph's global
behavior registry); `retrieve` wires the runtime to that registry and
fires `question.asked` to kick off the chain.

Determinism plumbing:
  - FrozenClock so timestamps are stable across re-ingests.
  - A fresh IDGen per ingest so object/event/relation counters start at
    zero — required for byte-identical event logs on re-ingest.
  - Fixed actor strings (`"ingest"`, `"agent.score_lexical"`, ...) and
    no explicit frame_id so provenance is identical across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from activegraph import Event, FrozenClock, Graph, IDGen, Runtime

# Importing this module registers the four agent behaviors into the
# activegraph global registry as a side effect. We capture the snapshot
# the first time so `retrieve` can re-register a clean slate per call.
from regimes.agent import behaviors as _behaviors_module  # noqa: F401
from regimes.agent import events as E
from regimes.agent.build import build_corpus


# Default LME-matching corpus knobs. Picked to match the reference
# "Mode A" config so graph content is comparable.
DEFAULT_MIN_TOKEN_LENGTH = 4
DEFAULT_MIN_SESSION_COOCCURRENCE = 2
DEFAULT_MAX_DOC_FREQ_FRACTION = 0.25
DEFAULT_TOKEN_BUDGET = 2500  # matches rag-dense-turn / rag-bm25-turn order

# Frozen ISO timestamp used for every ingest in this process. Tests
# override this via `FrozenClock(t=...)` indirectly through `retrieve`.
DEFAULT_FROZEN_T = "2026-01-01T00:00:00Z"

# Stable run_id stamped into every fresh Graph so provenance dicts are
# byte-identical across re-ingests. The real Runtime uses this only as a
# bookkeeping handle, not as a cross-run uniqueness constraint at our
# scale (no persistent SQLite store in the synchronous path); promotion's
# fork-and-diff path uses a different run_id derived from the fork.
DETERMINISTIC_RUN_ID = "regimes-agent-determ"


@dataclass
class AssembledContext:
    """The retrieval output. Matches the LME `AssembledContext` shape so the
    surrounding harness can consume it identically — text + truncated flag +
    meta dict. The meta carries everything the loop's diagnose step needs."""

    text: str
    truncated: bool
    meta: dict[str, Any]


@dataclass
class RetrieveTrace:
    """Per-question trace handle: the full event log + the assembled
    context. The loop reads `trace.events` for behavior-firing proof and
    `trace.context` for accuracy scoring."""

    context: AssembledContext
    events: list[Event]
    run_id: str
    ingest_stats: dict[str, Any]


# ----- the snapshot of "default agent behaviors" ----------------------------

# Importing `regimes.agent.behaviors` pushed our five @behavior decorations
# into activegraph's global _REGISTRY (lexical scorer, embedding scorer,
# transform pipeline, expand, assemble). Snapshot them once so we can
# restore them on every retrieve call (callers might have called
# clear_registry() between calls).
from activegraph import get_registry

_AGENT_BEHAVIORS_SNAPSHOT = [b for b in get_registry() if b.name.startswith("agent.")]


def _agent_behaviors() -> list:
    """Snapshot of the four agent behaviors. Callers pass this directly
    to `Runtime(graph, behaviors=...)` so the agent's runtime is
    independent of global-registry state."""
    return list(_AGENT_BEHAVIORS_SNAPSHOT)


def ingest(
    instance: dict[str, Any],
    *,
    min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    min_session_cooccurrence: int = DEFAULT_MIN_SESSION_COOCCURRENCE,
    max_doc_freq_fraction: float = DEFAULT_MAX_DOC_FREQ_FRACTION,
    frozen_t: str = DEFAULT_FROZEN_T,
) -> tuple[Graph, dict[str, Any]]:
    """Build the corpus graph for one LME instance using REAL package APIs.

    Returns (graph, ingest_stats). The graph is event-sourced; every
    Turn / Vocab / temporal / cooccurrence emission is a real event in
    `graph.events`.
    """
    graph = Graph(
        ids=IDGen(),
        clock=FrozenClock(frozen_t),
        run_id=DETERMINISTIC_RUN_ID,
    )
    res = build_corpus(
        graph,
        haystack_session_ids=instance["haystack_session_ids"],
        haystack_dates=instance["haystack_dates"],
        haystack_sessions=instance["haystack_sessions"],
        min_token_length=min_token_length,
        min_session_cooccurrence=min_session_cooccurrence,
        max_doc_freq_fraction=max_doc_freq_fraction,
    )
    stats = {
        "n_turns": res.n_turns,
        "n_temporal_edges": res.n_temporal_edges,
        "n_cooccurrence_edges": res.n_cooccurrence_edges,
        "vocab_object_id": res.vocab_object_id,
    }
    return graph, stats


DEFAULT_SIGNAL = "embedding"  # match the rag-dense comparison target


def retrieve(
    instance: dict[str, Any],
    *,
    question: str | None = None,
    question_date: str | None = None,
    signal: str = DEFAULT_SIGNAL,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    min_session_cooccurrence: int = DEFAULT_MIN_SESSION_COOCCURRENCE,
    max_doc_freq_fraction: float = DEFAULT_MAX_DOC_FREQ_FRACTION,
    frozen_t: str = DEFAULT_FROZEN_T,
) -> RetrieveTrace:
    """End-to-end: ingest then retrieve.

    Real package usage:
      - `Graph(ids=IDGen(), clock=FrozenClock(...))` — real Graph
      - `graph.add_object`, `graph.add_relation` — real mutators
      - `Runtime(graph, behaviors=[...])` — real runtime
      - `graph.emit(Event(...))` for the seed `question.asked`
      - `runtime.run_until_idle()` — real scheduler drives the chain
      - returns events read from `graph.events`
    """
    if signal not in ("lexical", "embedding"):
        from activegraph import ConfigurationError
        raise ConfigurationError(
            f"signal must be 'lexical' or 'embedding', got {signal!r}"
        )

    # Default the question fields from the instance shape if not given —
    # for the synthetic fixture and LME instances alike.
    q_text = question if question is not None else instance["question"]
    q_date = question_date if question_date is not None else instance.get("question_date", "")
    q_id = instance["question_id"]

    graph, ingest_stats = ingest(
        instance,
        min_token_length=min_token_length,
        min_session_cooccurrence=min_session_cooccurrence,
        max_doc_freq_fraction=max_doc_freq_fraction,
        frozen_t=frozen_t,
    )

    # Pass agent behaviors EXPLICITLY to Runtime so we don't depend on
    # the global behavior registry. The previous design did
    # `clear_registry() + register(_AGENT_BEHAVIORS_SNAPSHOT)` and
    # relied on `Runtime(graph)` resolving them from the global at
    # `_ensure_registry()` time — fragile when any caller (e.g. the
    # regimes loop, a test runner, a hot-reloader) clobbers the global
    # between install and run. With behaviors= the agent's runtime is
    # self-contained: it sees the snapshot we hand it, full stop.
    rt = Runtime(graph, behaviors=_AGENT_BEHAVIORS_SNAPSHOT)

    # Seed the chain with the real package emit path.
    seed = Event(
        id=graph.ids.event(),
        type=E.QUESTION_ASKED,
        payload={
            "question_id": q_id,
            "question": q_text,
            "question_date": q_date,
            "signal": signal,
            "token_budget": token_budget,
            "min_token_length": min_token_length,
        },
        actor="caller",
        caused_by=None,
        timestamp=graph.clock.now(),
    )
    graph.emit(seed)
    rt.run_until_idle()

    # Pluck the assembled context and the upstream events that carry
    # data the loop's diagnose step reads (ranked + post-transform scores
    # + applied transforms).
    assembled: Event | None = None
    expanded_ev: Event | None = None
    transformed_ev: Event | None = None
    for ev in reversed(graph.events):
        if ev.type == E.CONTEXT_ASSEMBLED and ev.payload.get("question_id") == q_id:
            assembled = ev
        if ev.type == E.TURNS_EXPANDED and ev.payload.get("question_id") == q_id:
            expanded_ev = ev
        if ev.type == E.TURNS_TRANSFORMED and ev.payload.get("question_id") == q_id:
            transformed_ev = ev
        if assembled and expanded_ev and transformed_ev:
            break

    # Scoring-step failures surface as behavior.failed events on
    # agent.score_lexical / agent.score_embedding. The loop's
    # scoring-error regime detector reads this; we lift it onto the meta
    # dict so a downstream Outcome can carry it without re-scanning the
    # event log.
    score_error_msg = ""
    for ev in graph.events:
        if ev.type == "behavior.failed":
            beh = ev.payload.get("behavior", "")
            if beh in ("agent.score_embedding", "agent.score_lexical"):
                etype = ev.payload.get("exception_type", "Error")
                msg = ev.payload.get("message", "")
                score_error_msg = f"{beh}:{etype}: {msg}"
                break

    if assembled is None:
        # The chain didn't reach assembly. The graph still holds the events
        # we did get — return an empty context with the trace, the loop's
        # failure-model handler turns this into a logged event upstream.
        ctx = AssembledContext(text="", truncated=True, meta={
            "error": "context.assembled missing",
            "score_error": score_error_msg,
            "n_events": len(graph.events),
            **ingest_stats,
        })
    else:
        p = assembled.payload
        ctx = AssembledContext(
            text=p["text"],
            truncated=p["truncated"],
            meta={
                "signal": signal,
                "n_selected_turns": len(p["selected_turn_ids"]),
                "n_seeds": p["n_seeds"],
                "n_expanded": p["n_expanded"],
                "token_budget": p["token_budget"],
                "running_tokens": p["running_tokens"],
                "selected_turn_ids": p["selected_turn_ids"],
                "decisions": p.get("decisions", []),
                "ranked": expanded_ev.payload["ranked"] if expanded_ev else [],
                "scores": transformed_ev.payload["scores"] if transformed_ev else {},
                "applied_transforms": (
                    transformed_ev.payload.get("applied_transforms", [])
                    if transformed_ev else []
                ),
                "score_error": score_error_msg,
                **ingest_stats,
            },
        )

    return RetrieveTrace(
        context=ctx,
        events=list(graph.events),
        run_id=rt.run_id,
        ingest_stats=ingest_stats,
    )
