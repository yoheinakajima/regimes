"""Top-level SQL-agent entrypoint: ingest the question's schema + draft
a SELECT.

This is the only place that constructs a Graph + Runtime for the SQL
agent. The four behaviors are imported (which side-effect-registers
them in activegraph's global behavior registry); `retrieve()` wires the
runtime to that snapshot and fires `question.asked` to kick off the
chain.

Determinism: same recipe as LME's agent — FrozenClock for timestamps,
fresh IDGen per ingest, stable run_id. The reader is held in a
per-question side-table in `behaviors.py` because Python callables
can't ride in event payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from activegraph import Event, FrozenClock, Graph, IDGen, Runtime, get_registry

# Importing this module registers the four sql_agent behaviors as a side
# effect. We capture the snapshot once at import time.
from regimes.targets.sql.agent import behaviors as _behaviors_module  # noqa: F401
from regimes.targets.sql.agent import events as E
from regimes.targets.sql.agent.behaviors import (
    _clear_reader,
    _set_reader,
)


DEFAULT_FROZEN_T = "2026-01-01T00:00:00Z"
DETERMINISTIC_RUN_ID = "regimes-sql-agent-determ"

_SQL_AGENT_BEHAVIORS_SNAPSHOT = [
    b for b in get_registry() if b.name.startswith("sql_agent.")
]


@dataclass
class DraftedQuery:
    """The agent's output for one question: the drafted SQL + the
    assembled prompt + the applied prompt_transforms + the schema_meta
    the seam saw."""

    predicted_sql: str
    drafter_error: str
    prompt: str
    prompt_parts: dict[str, Any]
    schema_meta: dict[str, Any]
    applied_transforms: tuple[str, ...]
    selected_column_ids: tuple[str, ...]
    column_scores: dict[str, float]


@dataclass
class RetrieveTrace:
    drafted: DraftedQuery
    events: list[Event]
    run_id: str


def retrieve(
    instance: dict[str, Any],
    *,
    reader: Any,
    frozen_t: str = DEFAULT_FROZEN_T,
) -> RetrieveTrace:
    """End-to-end: encode schema → score columns → assemble prompt
    (running prompt_transforms) → draft SQL via the Reader."""

    question_id = instance["question_id"]
    question = instance["question"]

    graph = Graph(
        ids=IDGen(),
        clock=FrozenClock(frozen_t),
        run_id=DETERMINISTIC_RUN_ID,
    )
    rt = Runtime(graph, behaviors=_SQL_AGENT_BEHAVIORS_SNAPSHOT)

    _set_reader(question_id, reader)
    try:
        seed = Event(
            id=graph.ids.event(),
            type=E.QUESTION_ASKED,
            payload={
                "question_id": question_id,
                "question": question,
                "schema_id": instance.get("schema_id", ""),
                "tables": list(instance["tables"]),
                "columns_by_table": {
                    k: list(v) for k, v in instance["columns_by_table"].items()
                },
                "foreign_keys": [list(fk) for fk in instance.get("foreign_keys", ())],
                "primary_keys": dict(instance.get("primary_keys", {})),
            },
            actor="caller",
            caused_by=None,
            timestamp=graph.clock.now(),
        )
        graph.emit(seed)
        rt.run_until_idle()

        prompt_ev: Event | None = None
        cols_ev: Event | None = None
        drafted_ev: Event | None = None
        for ev in reversed(graph.events):
            if ev.type == E.QUERY_DRAFTED and ev.payload.get("question_id") == question_id:
                drafted_ev = ev
            if ev.type == E.PROMPT_ASSEMBLED and ev.payload.get("question_id") == question_id:
                prompt_ev = ev
            if ev.type == E.COLUMNS_SCORED and ev.payload.get("question_id") == question_id:
                cols_ev = ev
            if drafted_ev and prompt_ev and cols_ev:
                break

        drafted = DraftedQuery(
            predicted_sql=(drafted_ev.payload["predicted_sql"] if drafted_ev else ""),
            drafter_error=(drafted_ev.payload.get("drafter_error", "") if drafted_ev else "chain_did_not_reach_draft"),
            prompt=(prompt_ev.payload["prompt"] if prompt_ev else ""),
            prompt_parts=(dict(prompt_ev.payload["prompt_parts"]) if prompt_ev else {}),
            schema_meta=(dict(prompt_ev.payload["schema_meta"]) if prompt_ev else {}),
            applied_transforms=tuple(
                prompt_ev.payload.get("applied_transforms", ()) if prompt_ev else ()
            ),
            selected_column_ids=tuple(
                cols_ev.payload.get("selected_column_ids", ()) if cols_ev else ()
            ),
            column_scores=dict(cols_ev.payload.get("scores", {}) if cols_ev else {}),
        )
        return RetrieveTrace(
            drafted=drafted,
            events=list(graph.events),
            run_id=rt.run_id,
        )
    finally:
        _clear_reader(question_id)
