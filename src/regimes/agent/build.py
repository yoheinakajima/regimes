"""Runtime-native graph construction.

Every graph mutation goes through the real activegraph package APIs:
  - turn nodes:        graph.add_object(type="turn", data={...})
  - corpus stats:      graph.add_object(type="vocab", data={...})
  - temporal edges:    graph.add_relation(type="temporal_next", ...)
  - co-occurrence:     graph.add_relation(type="cooccurrence", ...)

This module returns the LME turn_id -> activegraph object_id map so
callers (and the agent's behaviors) can resolve LME-style turn ids
against the runtime's typed object ids without re-scanning the graph.

The tokenization / vocab-pruning / edge-derivation logic matches the LME
reference (`activegraph_lme/activegraph/graph.py`); the difference is
that every state change is now an event on the real activegraph log,
not a tuple appended to an in-memory list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from activegraph import Graph

from regimes.agent.tokenize import count_tokens, distinctive_tokens, render_turn

INGEST_ACTOR = "ingest"


@dataclass
class IngestResult:
    """What ingest() returns to the agent. The graph itself owns the truth;
    this struct is the convenient handle for follow-up lookups."""

    graph: Graph
    turn_id_to_object_id: dict[str, str]
    vocab_object_id: str
    n_turns: int
    n_temporal_edges: int
    n_cooccurrence_edges: int


def build_corpus(
    graph: Graph,
    *,
    haystack_session_ids: list[str],
    haystack_dates: list[str],
    haystack_sessions: list[list[dict[str, Any]]],
    min_token_length: int,
    min_session_cooccurrence: int,
    max_doc_freq_fraction: float,
    actor: str = INGEST_ACTOR,
) -> IngestResult:
    """Emit a full LME haystack onto `graph` as real package events.

    Order is fixed and corpus-relative: sessions in input order, turns in
    input order, then vocab object, then temporal edges in session order,
    then co-occurrence edges in sorted (a,b) order. The fixed order is
    the contract that makes re-ingest produce a byte-identical event log
    when paired with a FrozenClock and a fresh IDGen.
    """
    turn_id_to_object_id: dict[str, str] = {}
    raw_tokens: dict[str, list[str]] = {}     # turn_id -> distinctive tokens (pre-prune)
    df_all: dict[str, int] = {}
    df_sessions: dict[str, set[str]] = {}
    turn_records: list[dict[str, Any]] = []   # ordered snapshot for edge pass

    # ---- 1) Materialize Turn objects (real object.created events) ----
    for s_idx, (sid, date, turns) in enumerate(
        zip(haystack_session_ids, haystack_dates, haystack_sessions)
    ):
        for t_idx, turn in enumerate(turns):
            role = str(turn.get("role", "?"))
            content = str(turn.get("content", ""))
            turn_id = f"{sid}#{t_idx}"
            text = render_turn(sid, date, role, content)

            toks = distinctive_tokens(content, min_token_length)
            raw_tokens[turn_id] = toks
            for tok in toks:
                df_all[tok] = df_all.get(tok, 0) + 1
                df_sessions.setdefault(tok, set()).add(sid)

            data = {
                "turn_id": turn_id,
                "session_id": sid,
                "session_date": date,
                "session_idx": s_idx,
                "turn_idx": t_idx,
                "role": role,
                "content": content,
                "text": text,
                # token_count and tokens are filled in after pruning;
                # we add them via a second add_object emit? No — provenance
                # would diverge. Instead, finalize before emit.
            }
            turn_records.append({"turn_id": turn_id, "data": data, "sid": sid, "t_idx": t_idx})

    # ---- 2) Prune vocab corpus-wide ----
    n_turns = max(1, len(turn_records))
    max_df = int(n_turns * max_doc_freq_fraction)
    kept: set[str] = set()
    for tok, df in df_all.items():
        if df > max_df:
            continue
        if len(df_sessions.get(tok, ())) < min_session_cooccurrence:
            continue
        kept.add(tok)

    # ---- 3) Emit Turn objects with pruned tokens and token_count ----
    for rec in turn_records:
        toks_kept = tuple(sorted(tok for tok in raw_tokens[rec["turn_id"]] if tok in kept))
        rec["data"]["tokens"] = list(toks_kept)
        rec["data"]["token_count"] = count_tokens(rec["data"]["text"])
        obj = graph.add_object(type="turn", data=rec["data"], actor=actor)
        turn_id_to_object_id[rec["turn_id"]] = obj.id

    # ---- 4) Emit Vocab object (corpus-level stats) ----
    vocab_data = {
        "df": {tok: df_all[tok] for tok in sorted(kept)},
        "n_turns": len(turn_records),
        "min_token_length": min_token_length,
        "min_session_cooccurrence": min_session_cooccurrence,
        "max_doc_freq_fraction": max_doc_freq_fraction,
    }
    vocab_obj = graph.add_object(type="vocab", data=vocab_data, actor=actor)

    # ---- 5) Temporal edges (within-session adjacency, input order) ----
    n_temporal = 0
    for s_idx, (sid, _, turns) in enumerate(
        zip(haystack_session_ids, haystack_dates, haystack_sessions)
    ):
        for t_idx in range(1, len(turns)):
            a_tid = f"{sid}#{t_idx - 1}"
            b_tid = f"{sid}#{t_idx}"
            a_obj = turn_id_to_object_id[a_tid]
            b_obj = turn_id_to_object_id[b_tid]
            graph.add_relation(
                source=a_obj,
                target=b_obj,
                type="temporal_next",
                data={"weight": 1.0},
                actor=actor,
            )
            n_temporal += 1

    # ---- 6) Co-occurrence edges (cross-session, on pruned vocab) ----
    tok_to_turn_ids: dict[str, list[str]] = {}
    for rec in turn_records:
        for tok in rec["data"]["tokens"]:
            tok_to_turn_ids.setdefault(tok, []).append(rec["turn_id"])

    turn_id_to_session = {rec["turn_id"]: rec["sid"] for rec in turn_records}
    pair_weight: dict[tuple[str, str], float] = {}
    for tok in sorted(tok_to_turn_ids):
        tids = tok_to_turn_ids[tok]
        df = max(1, vocab_data["df"][tok])
        w = 1.0 / df
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                if turn_id_to_session[a] == turn_id_to_session[b]:
                    continue
                key = (a, b) if a < b else (b, a)
                pair_weight[key] = pair_weight.get(key, 0.0) + w

    n_cooc = 0
    for a, b in sorted(pair_weight.keys()):
        wr = round(pair_weight[(a, b)], 9)
        graph.add_relation(
            source=turn_id_to_object_id[a],
            target=turn_id_to_object_id[b],
            type="cooccurrence",
            data={"weight": wr},
            actor=actor,
        )
        n_cooc += 1

    return IngestResult(
        graph=graph,
        turn_id_to_object_id=turn_id_to_object_id,
        vocab_object_id=vocab_obj.id,
        n_turns=len(turn_records),
        n_temporal_edges=n_temporal,
        n_cooccurrence_edges=n_cooc,
    )
