"""The four SQL-agent behaviors as real `@activegraph.behavior`
registrations.

Mirrors `regimes.agent.behaviors` for LongMemEval — same pattern:
behaviors are decorated at import time, the snapshot is captured in
`agent.py`, and the Runtime is constructed with `behaviors=` pinned to
that snapshot so the global registry can't leak between agent runs.

Per-question chain:
    question.asked
      -> behavior_encode_schema   -> schema.encoded
      -> behavior_score_columns   -> columns.scored
      -> behavior_prompt_pipeline -> prompt.assembled
      -> behavior_draft_query     -> query.drafted

The `reader` (Fake or Anthropic) is passed via the seed event's payload
on a process-level context dict, the same indirection LME's loop uses
for its Python callables."""

from __future__ import annotations

import math
import threading
from typing import Any

from activegraph import behavior

from regimes.agent.embedders import get_embedder
from regimes.targets.sql import prompt_transforms
from regimes.targets.sql.agent import events as E


# ===========================================================================
# Reader-context indirection
# ===========================================================================
# Python callables don't survive event payloads. The retrieve() entry
# point sets the reader here keyed by question_id; behavior_draft_query
# reads it back.

_READERS: dict[str, Any] = {}
_READER_LOCK = threading.Lock()


def _set_reader(question_id: str, reader: Any) -> None:
    with _READER_LOCK:
        _READERS[question_id] = reader


def _get_reader(question_id: str) -> Any:
    with _READER_LOCK:
        return _READERS.get(question_id)


def _clear_reader(question_id: str) -> None:
    with _READER_LOCK:
        _READERS.pop(question_id, None)


# ===========================================================================
# 1) Encode the schema as graph objects + relations
# ===========================================================================

@behavior(name="sql_agent.encode_schema", on=[E.QUESTION_ASKED])
def behavior_encode_schema(event, graph, ctx) -> None:  # noqa: ARG001
    """Each column → one `column` object with data {table, name, type,
    is_pk}. Each foreign key → one `foreign_key` relation from the FK
    column object to the referenced column object. Each table → one
    `table` object so the prompt-assembler can iterate them cheaply."""
    payload = event.payload
    question_id = payload["question_id"]
    tables: list[str] = list(payload["tables"])
    columns_by_table: dict[str, list[str]] = dict(payload["columns_by_table"])
    foreign_keys: list[tuple[str, str, str, str]] = [
        tuple(fk) for fk in payload["foreign_keys"]
    ]
    pk_columns: dict[str, str] = dict(payload.get("primary_keys", {}))

    table_obj_id: dict[str, str] = {}
    col_obj_id: dict[tuple[str, str], str] = {}

    # Tables first, in input order.
    for tname in tables:
        o = graph.add_object(type="table", data={"name": tname})
        table_obj_id[tname] = o.id

    # Then columns, table-by-table.
    for tname in tables:
        for cname in columns_by_table.get(tname, ()):
            o = graph.add_object(
                type="column",
                data={
                    "table": tname,
                    "name": cname,
                    "qualified": f"{tname}.{cname}",
                    "is_pk": (pk_columns.get(tname) == cname),
                },
            )
            col_obj_id[(tname, cname)] = o.id

    # FK relations.
    n_fk = 0
    for (t, c, rt, rc) in foreign_keys:
        src = col_obj_id.get((t, c))
        dst = col_obj_id.get((rt, rc))
        if src and dst:
            graph.add_relation(
                source=src,
                target=dst,
                type="foreign_key",
                data={"from": f"{t}.{c}", "to": f"{rt}.{rc}"},
            )
            n_fk += 1

    graph.emit(
        E.SCHEMA_ENCODED,
        {
            "question_id": question_id,
            "question": payload["question"],
            "schema_id": payload.get("schema_id", ""),
            "n_tables": len(tables),
            "n_columns": sum(len(v) for v in columns_by_table.values()),
            "n_foreign_keys": n_fk,
        },
    )


# ===========================================================================
# 2) Embedding-score columns against the question; top-k
# ===========================================================================

@behavior(name="sql_agent.retrieve_relevant_columns", on=[E.SCHEMA_ENCODED])
def behavior_score_columns(event, graph, ctx) -> None:  # noqa: ARG001
    """Cosine-similarity between the question and each column's textual
    representation (`table.column_name`). Top-K becomes the
    `selected_column_ids`. Same HashEmbedder LME uses by default."""
    payload = event.payload
    question_id = payload["question_id"]
    question = payload["question"]
    top_k = 12  # default seed window; small schemas mean this rarely binds

    embedder = get_embedder()

    cols: list[tuple[str, str, str]] = []  # (object_id, qualified, table)
    for o in ctx.view.objects(type="column"):
        cols.append((o.id, o.data["qualified"], o.data["table"]))

    texts = [question] + [q for _, q, _ in cols]
    vecs = embedder.embed(texts)
    q_vec = vecs[0]

    scores: dict[str, float] = {}
    for (oid, qual, _), v in zip(cols, vecs[1:]):
        scores[qual] = _cosine(q_vec, v)

    ranked = sorted(scores.keys(), key=lambda q: -scores[q])
    selected = ranked[:top_k]

    graph.emit(
        E.COLUMNS_SCORED,
        {
            "question_id": question_id,
            "question": question,
            "scorer_model": embedder.model,
            "scores": scores,
            "ranked": ranked,
            "selected_column_ids": selected,
        },
    )


def _cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    # vectors come back L2-normalized from HashEmbedder
    return s


# ===========================================================================
# 3) Prompt-pipeline seam (the optimization target)
# ===========================================================================

@behavior(name="sql_agent.prompt_pipeline", on=[E.COLUMNS_SCORED])
def behavior_prompt_pipeline(event, graph, ctx) -> None:  # noqa: ARG001
    """Assemble the four `prompt_parts` (schema, instructions, hints,
    question), then run the configurable `prompt_transforms` pipeline,
    then render the final prompt string and emit it.

    Default pipeline is empty → passthrough. The pipeline contents are
    recorded as `applied_transforms` in the emitted event."""
    payload = event.payload
    question_id = payload["question_id"]
    question = payload["question"]
    selected: list[str] = payload["selected_column_ids"]

    # Build a compact schema text from the selected columns + their
    # parent tables.
    cols_by_table: dict[str, list[str]] = {}
    for o in ctx.view.objects(type="column"):
        qual = o.data["qualified"]
        if qual not in selected:
            continue
        cols_by_table.setdefault(o.data["table"], []).append(o.data["name"])

    # Foreign-key hints from the graph: enumerate fk relations whose
    # endpoints are both in the selected set so the LLM sees the join
    # paths it can use.
    qual_in_selected = set(selected)
    fk_hints: list[str] = []
    for r in ctx.view.relations(type="foreign_key"):
        fr = r.data.get("from", "")
        to = r.data.get("to", "")
        if fr in qual_in_selected and to in qual_in_selected:
            fk_hints.append(f"{fr} -> {to}")

    schema_text_lines: list[str] = []
    for t, cols in sorted(cols_by_table.items()):
        schema_text_lines.append(f"  {t}({', '.join(cols)})")
    schema_text = "Tables:\n" + "\n".join(schema_text_lines) if schema_text_lines else "(no schema)"

    schema_meta = {
        "tables": list(sorted(cols_by_table)),
        "columns_by_table": {k: list(v) for k, v in cols_by_table.items()},
        "foreign_keys": fk_hints,
    }

    prompt_parts: dict[str, Any] = {
        "schema": schema_text,
        "instructions": (
            "Write a single SQLite SELECT statement that answers the question. "
            "Return only the SQL — no prose."
        ),
        "hints": list(fk_hints),
        "question": question,
    }

    applied, errors = prompt_transforms.apply_pipeline(
        prompt_parts=prompt_parts,
        question=question,
        schema_meta=schema_meta,
    )
    final_parts = applied["prompt_parts"]

    # Assemble the final prompt string.
    hints_block = ""
    if final_parts.get("hints"):
        hints_block = "Hints:\n  " + "\n  ".join(final_parts["hints"]) + "\n"
    final_prompt = (
        f"{final_parts['schema']}\n\n"
        f"{hints_block}"
        f"Instructions: {final_parts['instructions']}\n\n"
        f"Question: {final_parts['question']}\n\n"
        f"SQL:"
    )

    graph.emit(
        E.PROMPT_ASSEMBLED,
        {
            "question_id": question_id,
            "question": question,
            "prompt": final_prompt,
            "prompt_parts": final_parts,
            "schema_meta": schema_meta,
            "applied_transforms": applied["names"],
            "transform_errors": errors,
        },
    )


# ===========================================================================
# 4) Draft the SQL via the Reader (Fake or Anthropic)
# ===========================================================================

@behavior(name="sql_agent.draft_query", on=[E.PROMPT_ASSEMBLED])
def behavior_draft_query(event, graph, ctx) -> None:  # noqa: ARG001
    """Call the configured Reader to draft a SELECT. Failures are
    recorded on the event as `drafter_error` (per the framework's
    failure model — errors during runtime become event payload
    entries, not raises)."""
    payload = event.payload
    question_id = payload["question_id"]
    question = payload["question"]
    prompt = payload["prompt"]

    reader = _get_reader(question_id)
    drafter_error = ""
    sql = ""
    if reader is None:
        drafter_error = "reader_missing: no Reader registered for question_id"
    else:
        try:
            sql = reader.answer(context=prompt, question=question, question_id=question_id)
        except Exception as e:  # noqa: BLE001 — runtime path
            drafter_error = f"{type(e).__name__}: {e}"

    graph.emit(
        E.QUERY_DRAFTED,
        {
            "question_id": question_id,
            "predicted_sql": sql.strip() if isinstance(sql, str) else "",
            "drafter_error": drafter_error,
            "reader_name": getattr(reader, "name", "") if reader else "",
        },
    )
