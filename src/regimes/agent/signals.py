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
import traceback as _traceback
from typing import Any, Callable

from regimes.agent.embedders import Embedder, embedding_token_count
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


def _coerce_text(value: Any) -> str:
    """Embedder inputs must be strings. Coerce None / non-str defensively
    so a malformed turn can't blow up the whole batch with a TypeError
    inside the embedder."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _embedding_error_record(turn_id: str, text: str, exc: BaseException) -> dict[str, Any]:
    """An auditable record of one failed embedding input: WHAT failed
    (turn_id), the shape of the offending input (chars + token count + a
    truncated repr), and WHY (exception type/message + full traceback)."""
    return {
        "turn_id": turn_id,
        "input_chars": len(text),
        "input_tokens": embedding_token_count(text),
        "input_repr": repr(text[:200]),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            _traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _embed_resilient(
    embedder: Embedder,
    texts: list[str],
    ids: list[str],
    *,
    on_error: Callable[[dict[str, Any]], None] | None,
) -> list[list[float]]:
    """Embed `texts`, isolating and reporting any offending input.

    Fast path: one batched `embed()` call. If that raises, fall back to
    per-input embedding so the exact failing turn(s) can be identified;
    failing inputs are replaced with a zero vector (cosine 0.0 against
    anything — a neutral sentinel that won't corrupt other turns' scores)
    and reported via `on_error`. The signal degrades gracefully and
    audibly instead of taking the whole question down."""
    try:
        return embedder.embed(list(texts))
    except Exception:  # noqa: BLE001 — re-isolated per input below
        vecs: list[list[float] | None] = []
        dim: int | None = None
        for tid, text in zip(ids, texts):
            try:
                v = embedder.embed([text])[0]
                if dim is None and v:
                    dim = len(v)
                vecs.append(list(v))
            except Exception as exc:  # noqa: BLE001 — reported, not swallowed
                if on_error is not None:
                    on_error(_embedding_error_record(tid, text, exc))
                vecs.append(None)
        zero = [0.0] * (dim or 1)
        return [v if v is not None else list(zero) for v in vecs]


def score_embedding(
    view: Any,
    question: str,
    embedder: Embedder,
    *,
    on_error: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    """Cosine similarity between question and each turn's text.

    Matches LME reference: L2-normalized vectors, so cosine reduces to a
    dot product. Returns a dense dict over every turn.

    `on_error`, when provided, is invoked with an auditable record (see
    `_embedding_error_record`) for every input the embedder rejects. The
    offending turn gets a neutral zero-vector score rather than crashing
    the scoring step; the caller decides how to log/persist the record.
    """
    turns = list(view.objects(type="turn"))
    if not turns:
        return {}

    turn_ids = [t.data["turn_id"] for t in turns]
    texts = [_coerce_text(t.data.get("text")) for t in turns]
    turn_vecs = _embed_resilient(embedder, texts, turn_ids, on_error=on_error)
    q_vec = _embed_resilient(
        embedder, [_coerce_text(question)], ["<question>"], on_error=on_error
    )[0]

    scores: dict[str, float] = {}
    for tid, vec in zip(turn_ids, turn_vecs):
        s = 0.0
        for x, y in zip(vec, q_vec):
            s += x * y
        scores[tid] = float(s)
    return scores
