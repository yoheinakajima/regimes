"""Embedders for the embedding scoring signal.

Two implementations behind one protocol:

  HashEmbedder    — deterministic, no network, no API keys. Used for unit
                    tests and for any environment without OpenAI access.
                    Vectors are SHA-256 / sign-hashing into a fixed-dim
                    space, then L2-normalized.

  OpenAIEmbedder  — text-embedding-3-small via the openai SDK. Matches the
                    LME reference embedder exactly. Requires
                    OPENAI_API_KEY in the environment and the `openai`
                    package installed. Construction raises
                    activegraph.ConfigurationError if either is missing
                    (caller-fixable construction error per the failure
                    model).

Both implementations:
  - take `embed(texts: list[str]) -> list[list[float]]` (L2-normalized)
  - carry a stable `model` string so the audit log knows which embedder
    produced any given score

The active embedder is a process-level singleton set by
`set_embedder(...)`. `get_embedder()` returns the current one (defaults
to HashEmbedder on first access so tests work out of the box).
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from typing import Protocol, runtime_checkable

from activegraph import ConfigurationError


@runtime_checkable
class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ----- HashEmbedder (deterministic, no network) -----------------------------

class HashEmbedder:
    """Deterministic embedder. NOT semantically meaningful — it's a
    feature-hashed bag-of-tokens projection. Used for unit tests and the
    synthetic fixture so the embedding path is exercised end-to-end with
    no API keys.

    Determinism: pure function of input text bytes. Re-running on the
    same corpus produces byte-identical vectors.
    """

    model = "regimes.hash-embedder-v1"

    def __init__(self, dim: int = 64) -> None:
        if dim < 8:
            raise ConfigurationError(f"HashEmbedder dim must be >= 8, got {dim}")
        self._dim = dim
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        with self._lock:
            for t in texts:
                v = self._cache.get(t)
                if v is None:
                    v = self._encode_one(t)
                    self._cache[t] = v
                out.append(v)
        return out

    def _encode_one(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        # Use the same tokenization as the lexical signal so the two
        # signals see the corpus the same way — keeps comparisons clean.
        from regimes.agent.tokenize import raw_tokenize
        for tok in raw_tokenize(text):
            h_bytes = hashlib.sha256(tok.encode("utf-8")).digest()
            h = int.from_bytes(h_bytes[:8], "big")
            idx = h % self._dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            v[idx] += sign
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v


# ----- OpenAIEmbedder (LME-pinned production path) ---------------------------

class OpenAIEmbedder:
    """text-embedding-3-small via openai. Matches the LME reference.

    Construction validates env + import; embed() is called lazily.
    No cache flushing across processes; cache is per-instance.
    """

    model = "text-embedding-3-small"

    def __init__(self, *, batch_size: int = 96) -> None:
        if "OPENAI_API_KEY" not in os.environ:
            raise ConfigurationError(
                "OpenAIEmbedder requires OPENAI_API_KEY in the environment."
            )
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:
            raise ConfigurationError(
                "OpenAIEmbedder requires the `openai` package. "
                "Install: pip install openai"
            ) from e
        self._batch_size = batch_size
        self._client = None
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _ensure_client(self):  # pragma: no cover — network path
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        cli = self._ensure_client()
        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        to_fetch: list[tuple[int, str]] = []
        with self._lock:
            for i, t in enumerate(texts):
                v = self._cache.get(t)
                if v is not None:
                    out[i] = v
                else:
                    to_fetch.append((i, t))
        if to_fetch:
            new_texts = [t for _, t in to_fetch]
            new_vecs: list[list[float]] = []
            for j in range(0, len(new_texts), self._batch_size):
                resp = cli.embeddings.create(model=self.model, input=new_texts[j : j + self._batch_size])
                for d in resp.data:
                    vec = list(d.embedding)
                    n = math.sqrt(sum(x * x for x in vec))
                    new_vecs.append([x / n for x in vec] if n > 0 else vec)
            with self._lock:
                for (i, t), v in zip(to_fetch, new_vecs):
                    self._cache[t] = v
                    out[i] = v
        return out


# ----- process-level singleton ----------------------------------------------

_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def set_embedder(embedder: Embedder) -> None:
    """Replace the process embedder. Production wiring calls this once at
    startup with OpenAIEmbedder()."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        _EMBEDDER = embedder


def get_embedder() -> Embedder:
    """Return the current embedder. Defaults to HashEmbedder so tests and
    synthetic-fixture runs work with no extra setup."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = HashEmbedder()
        return _EMBEDDER


def reset_embedder() -> None:
    """Drop the configured embedder; next get_embedder() returns a fresh
    HashEmbedder. Used for test isolation."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        _EMBEDDER = None
