"""Pure scoring math. Each signal takes (graph, question, min_token_length)
and returns a dict[turn_id -> float].

These functions are called from inside @behavior bodies; they do not
emit events themselves. Keeping them pure makes them trivially testable
and lets a property test compare bit-for-bit against the LME reference.

`score_lexical` reads:
  - every Turn object (graph.objects(type="turn")) for data["tokens"]
  - the singleton Vocab object for data["df"] and data["n_turns"]

Both are real activegraph reads through the package's query API
(`graph.objects(type=...)`), not a side projection.
"""

from __future__ import annotations

import math

from activegraph import Graph

from regimes.agent.tokenize import distinctive_tokens


def _vocab(graph: Graph) -> dict:
    candidates = graph.objects(type="vocab")
    if not candidates:
        return {"df": {}, "n_turns": 0}
    if len(candidates) > 1:
        # Defensive: the agent only ever emits one vocab per ingest. Multiple
        # would mean stacked ingests on one runtime, which the agent doesn't
        # do. Use the most recently added one (last in projection order).
        return candidates[-1].data
    return candidates[0].data


def score_lexical(graph: Graph, question: str, min_token_length: int) -> dict[str, float]:
    """IDF-weighted distinctive-token overlap.

    Definition (matches LME reference spec):
      idf(t) = log((n_turns + 1) / (df(t) + 1)) + 1.0   smoothed, > 0
      score(turn) = sum_{t in turn.tokens n question.tokens} idf(t)

    Tokens outside the pruned vocab contribute 0. Returns a dense dict
    over every turn (turns with no overlap get 0.0).
    """
    q_tokens = distinctive_tokens(question, min_token_length)
    vocab = _vocab(graph)
    df_map: dict[str, int] = vocab.get("df", {})
    n_turns = max(1, int(vocab.get("n_turns", 0)))

    idf: dict[str, float] = {}
    for tok in q_tokens:
        df = df_map.get(tok)
        if df is None:
            continue
        idf[tok] = math.log((n_turns + 1) / (df + 1)) + 1.0

    scores: dict[str, float] = {}
    for t_obj in graph.objects(type="turn"):
        turn_id = t_obj.data["turn_id"]
        toks = t_obj.data.get("tokens", ())
        if not toks or not idf:
            scores[turn_id] = 0.0
            continue
        s = 0.0
        for tok in toks:
            w = idf.get(tok)
            if w is not None:
                s += w
        scores[turn_id] = s
    return scores
