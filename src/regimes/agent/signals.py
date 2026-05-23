"""Pure scoring math. Each signal takes a read-only graph view + the
question and returns a dict[turn_id -> float].

These functions are called from inside @behavior bodies; they do not
emit events themselves. Keeping them pure makes them trivially testable.

`score_lexical` reads:
  - every Turn object's data["tokens"]
  - the singleton Vocab object's data["df"] and data["n_turns"]

`score_embedding` reads:
  - every Turn object's data["text"]
  - calls the configured Embedder once for all turns + once for the question,
    computes cosine similarity (= dot product on L2-normalized vectors)

Both read through the package's query API (`view.objects(type=...)`),
not a side projection.
"""

from __future__ import annotations

import math
from typing import Any

from regimes.agent.embedders import Embedder
from regimes.agent.tokenize import distinctive_tokens


def _vocab(view: Any) -> dict:
    candidates = view.objects(type="vocab")
    if not candidates:
        return {"df": {}, "n_turns": 0}
    if len(candidates) > 1:
        return candidates[-1].data
    return candidates[0].data


def score_lexical(view: Any, question: str, min_token_length: int) -> dict[str, float]:
    """IDF-weighted distinctive-token overlap.

    Definition (matches LME reference spec):
      idf(t) = log((n_turns + 1) / (df(t) + 1)) + 1.0   smoothed, > 0
      score(turn) = sum_{t in turn.tokens n question.tokens} idf(t)
    """
    q_tokens = distinctive_tokens(question, min_token_length)
    vocab = _vocab(view)
    df_map: dict[str, int] = vocab.get("df", {})
    n_turns = max(1, int(vocab.get("n_turns", 0)))

    idf: dict[str, float] = {}
    for tok in q_tokens:
        df = df_map.get(tok)
        if df is None:
            continue
        idf[tok] = math.log((n_turns + 1) / (df + 1)) + 1.0

    scores: dict[str, float] = {}
    for t_obj in view.objects(type="turn"):
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


def score_embedding(
    view: Any,
    question: str,
    embedder: Embedder,
) -> dict[str, float]:
    """Cosine similarity between question and each turn's text.

    Matches LME reference: L2-normalized vectors, so cosine reduces to a
    dot product. Returns a dense dict over every turn.
    """
    turns = view.objects(type="turn")
    if not turns:
        return {}

    texts = [t.data["text"] for t in turns]
    turn_vecs = embedder.embed(texts)
    q_vec = embedder.embed([question])[0]

    scores: dict[str, float] = {}
    for t_obj, vec in zip(turns, turn_vecs):
        s = 0.0
        for x, y in zip(vec, q_vec):
            s += x * y
        scores[t_obj.data["turn_id"]] = float(s)
    return scores
