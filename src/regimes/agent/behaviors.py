"""The four agent behaviors as real @behavior registrations.

Each is decorated with `activegraph.behavior` (the package decorator) so
the package's registry holds them and the runtime fires them via its
own scheduler.

Per the package contract (CONTRACT #7, #11):
  - the `graph` arg is a BehaviorGraph (mutators + emit only)
  - iteration goes through `ctx.view.objects(type=...)` and
    `ctx.view.relations(type=...)`
  - custom events are emitted via `graph.emit(type_str, payload_dict)` —
    the runtime stamps actor/caused_by/frame_id automatically

Event chain (one question):
    question.asked
      -> behavior_score   -> turns.scored
      -> behavior_transform -> turns.transformed
      -> behavior_expand  -> turns.expanded
      -> behavior_assemble -> context.assembled
"""

from __future__ import annotations

from typing import Any

from activegraph import behavior

from regimes.agent import events as E
from regimes.agent import transforms
from regimes.agent.signals import score_lexical


# ----- 1) Scoring ------------------------------------------------------------

@behavior(name="agent.score_lexical", on=[E.QUESTION_ASKED])
def behavior_score(event, graph, ctx) -> None:
    """Read corpus through ctx.view, score every turn, emit turns.scored."""
    payload = event.payload
    question = payload["question"]
    question_id = payload["question_id"]
    min_token_length = payload["min_token_length"]
    token_budget = payload["token_budget"]

    scores = score_lexical(ctx.view, question, min_token_length=min_token_length)

    graph.emit(
        E.TURNS_SCORED,
        {
            "question_id": question_id,
            "question": question,
            "question_date": payload.get("question_date", ""),
            "signal": "lexical",
            "scores": scores,
            "token_budget": token_budget,
            "min_token_length": min_token_length,
        },
    )


# ----- 2) Transform seam -----------------------------------------------------

@behavior(name="agent.transform_pipeline", on=[E.TURNS_SCORED])
def behavior_transform(event, graph, ctx) -> None:
    """Apply the configured score-transform pipeline; emit turns.transformed.

    Default pipeline is empty → passthrough. The pipeline contents are
    recorded as `applied_transforms` in the emitted event.
    """
    payload = event.payload
    scores = dict(payload["scores"])
    question_id = payload["question_id"]
    token_budget = payload["token_budget"]
    min_token_length = payload["min_token_length"]
    question = payload.get("question", "")
    question_date = payload.get("question_date", "")

    applied, errors = transforms.apply_pipeline(
        scores=scores,
        graph=ctx.view,  # transforms see the read-only view, not the bgraph
        question_id=question_id,
        question=question,
        question_date=question_date,
    )

    graph.emit(
        E.TURNS_TRANSFORMED,
        {
            "question_id": question_id,
            "question": question,
            "question_date": question_date,
            "scores": applied["scores"],
            "applied_transforms": applied["names"],
            "transform_errors": errors,
            "token_budget": token_budget,
            "min_token_length": min_token_length,
        },
    )


# ----- 3) Temporal expansion -------------------------------------------------

@behavior(name="agent.expand_temporal", on=[E.TURNS_TRANSFORMED])
def behavior_expand(event, graph, ctx) -> None:
    """Walk real `temporal_next` relations via ctx.view.

    The View API doesn't take source/target filters on relations, so we
    pull every temporal_next relation once and build outgoing/incoming
    indices in Python. For corpora of <500 turns this is O(n) and cheap.
    """
    payload = event.payload
    scores: dict[str, float] = payload["scores"]
    question_id = payload["question_id"]
    token_budget = payload["token_budget"]

    # Build turn_id <-> object_id and chronological sort key.
    tid_to_obj: dict[str, str] = {}
    obj_to_tid: dict[str, str] = {}
    sort_key: dict[str, tuple] = {}
    for o in ctx.view.objects(type="turn"):
        tid = o.data["turn_id"]
        tid_to_obj[tid] = o.id
        obj_to_tid[o.id] = tid
        sort_key[tid] = (
            o.data["session_date"],
            o.data["session_idx"],
            o.data["turn_idx"],
        )

    # Pull temporal_next once; index by source and by target.
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for r in ctx.view.relations(type="temporal_next"):
        outgoing.setdefault(r.source, []).append(r.target)
        incoming.setdefault(r.target, []).append(r.source)

    # Ranked: -score desc, chronological asc.
    ranked = sorted(
        sort_key.keys(),
        key=lambda tid: (-scores.get(tid, 0.0), sort_key[tid]),
    )

    expansion_map: dict[str, list[str]] = {}
    for tid in ranked:
        if scores.get(tid, 0.0) <= 0.0:
            continue
        obj_id = tid_to_obj[tid]
        neighbors: list[str] = []
        for tgt_obj in outgoing.get(obj_id, ()):
            n_tid = obj_to_tid.get(tgt_obj)
            if n_tid is not None:
                neighbors.append(n_tid)
        for src_obj in incoming.get(obj_id, ()):
            n_tid = obj_to_tid.get(src_obj)
            if n_tid is not None:
                neighbors.append(n_tid)
        if neighbors:
            expansion_map[tid] = neighbors

    graph.emit(
        E.TURNS_EXPANDED,
        {
            "question_id": question_id,
            "ranked": ranked,
            "expansion_map": expansion_map,
            "scores": scores,
            "token_budget": token_budget,
        },
    )


# ----- 4) Greedy budget assembly --------------------------------------------

@behavior(name="agent.assemble", on=[E.TURNS_EXPANDED])
def behavior_assemble(event, graph, ctx) -> None:
    """Greedy budget walk: pick seeds by score under budget, then add
    their temporal neighbors under budget, then emit the assembled
    context in chronological order. One rich event."""
    payload = event.payload
    scores: dict[str, float] = payload["scores"]
    ranked: list[str] = payload["ranked"]
    expansion_map: dict[str, list[str]] = payload["expansion_map"]
    token_budget: int = payload["token_budget"]
    question_id = payload["question_id"]

    by_tid: dict[str, dict[str, Any]] = {}
    sort_key: dict[str, tuple] = {}
    for o in ctx.view.objects(type="turn"):
        tid = o.data["turn_id"]
        by_tid[tid] = {
            "text": o.data["text"],
            "token_count": int(o.data["token_count"]),
        }
        sort_key[tid] = (
            o.data["session_date"],
            o.data["session_idx"],
            o.data["turn_idx"],
        )

    selected: list[str] = []
    selected_set: set[str] = set()
    running = 0
    truncated = False
    decisions: list[dict[str, Any]] = []
    n_seeds = 0

    def _cost(tid: str) -> int:
        return by_tid[tid]["token_count"] + 2  # +2 approximates "\n\n" joiner

    def _try_add(tid: str, reason_in: str) -> bool:
        nonlocal running, truncated
        if tid in selected_set:
            return True
        if tid not in by_tid:
            decisions.append(
                {"turn_id": tid, "score": scores.get(tid, 0.0),
                 "included": False, "reason": "unknown_turn"}
            )
            return False
        n = _cost(tid)
        if running + n > token_budget:
            truncated = True
            decisions.append(
                {"turn_id": tid, "score": scores.get(tid, 0.0),
                 "included": False, "reason": "budget"}
            )
            return False
        selected.append(tid)
        selected_set.add(tid)
        running += n
        decisions.append(
            {"turn_id": tid, "score": scores.get(tid, 0.0),
             "included": True, "reason": reason_in}
        )
        return True

    # ---- seed selection by score ----
    for tid in ranked:
        if scores.get(tid, 0.0) <= 0.0 and n_seeds > 0:
            decisions.append(
                {"turn_id": tid, "score": scores.get(tid, 0.0),
                 "included": False, "reason": "zero_score_post_seed"}
            )
            break
        if _try_add(tid, reason_in="seed"):
            n_seeds += 1

    # ---- 1-hop expansion over selected seeds ----
    n_expanded = 0
    expansion_targets: list[str] = []
    seeds_snapshot = list(selected)
    for tid in seeds_snapshot:
        for n_tid in expansion_map.get(tid, ()):
            if n_tid not in selected_set:
                expansion_targets.append(n_tid)

    expansion_targets = sorted(
        set(expansion_targets),
        key=lambda t: sort_key.get(t, ("", 0, 0)),
    )
    for tid in expansion_targets:
        if _try_add(tid, reason_in="expand"):
            n_expanded += 1

    # chronological emit order
    selected.sort(key=lambda t: sort_key[t])
    assembled_text = "\n\n".join(by_tid[t]["text"] for t in selected)

    graph.emit(
        E.CONTEXT_ASSEMBLED,
        {
            "question_id": question_id,
            "selected_turn_ids": selected,
            "n_seeds": n_seeds,
            "n_expanded": n_expanded,
            "truncated": truncated,
            "token_budget": token_budget,
            "running_tokens": running,
            "text": assembled_text,
            "decisions": decisions,
        },
    )
