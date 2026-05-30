"""Embedding scoring path: gating, determinism, and parity-of-shape.

Tests prove:
  - the agent.score_embedding behavior fires when signal="embedding"
  - the agent.score_lexical behavior does NOT fire when signal="embedding"
    (where={"signal": ...} gating works)
  - HashEmbedder produces deterministic scores (same input → same vec)
  - the embedding path produces the same downstream event shape as
    lexical (turns.transformed / turns.expanded / context.assembled)
  - OpenAIEmbedder fails cleanly with ConfigurationError when keys/imports
    are missing — the production wiring's failure mode is visible
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from activegraph import ConfigurationError

from regimes.agent import HashEmbedder, OpenAIEmbedder, retrieve, set_embedder

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "synthetic_lme.json"


@pytest.fixture(scope="module")
def instances():
    return json.loads(FIXTURE.read_text())


def _scored_event(events):
    for e in events:
        if e.type == "turns.scored":
            return e
    raise AssertionError("no turns.scored event")


def test_embedding_signal_fires_correct_behavior(instances):
    t = retrieve(instances[0], signal="embedding", token_budget=2500)
    scored = _scored_event(t.events)
    assert scored.actor == "agent.score_embedding"
    assert scored.payload["signal"] == "embedding"
    assert scored.payload["scorer_model"] == "regimes.hash-embedder-v1"


def test_lexical_signal_fires_correct_behavior(instances):
    t = retrieve(instances[0], signal="lexical", token_budget=2500)
    scored = _scored_event(t.events)
    assert scored.actor == "agent.score_lexical"
    assert scored.payload["signal"] == "lexical"


def test_only_one_scoring_behavior_fires_per_question(instances):
    """The where={"signal": ...} gating means only the matching scorer
    fires; if both fired, we'd see two turns.scored events and two
    parallel downstream chains."""
    for sig in ("lexical", "embedding"):
        t = retrieve(instances[0], signal=sig, token_budget=2500)
        scored_events = [e for e in t.events if e.type == "turns.scored"]
        assert len(scored_events) == 1, (
            f"signal={sig}: expected 1 turns.scored, got {len(scored_events)}"
        )
        assembled = [e for e in t.events if e.type == "context.assembled"]
        assert len(assembled) == 1


def test_embedding_determinism(instances):
    t1 = retrieve(instances[0], signal="embedding", token_budget=2500)
    t2 = retrieve(instances[0], signal="embedding", token_budget=2500)
    s1 = _scored_event(t1.events).payload["scores"]
    s2 = _scored_event(t2.events).payload["scores"]
    assert s1 == s2, "embedding scores diverged across runs"
    assert t1.context.text == t2.context.text


def test_embedding_path_same_event_shape_as_lexical(instances):
    """Both signals produce the same downstream event types in the same
    order — the transform seam, expansion, and assembly are signal-agnostic."""
    t_emb = retrieve(instances[0], signal="embedding", token_budget=2500)
    t_lex = retrieve(instances[0], signal="lexical", token_budget=2500)
    chain_types = ["question.asked", "turns.scored", "turns.transformed",
                   "turns.expanded", "context.assembled"]
    emb_chain = [e.type for e in t_emb.events if e.type in chain_types]
    lex_chain = [e.type for e in t_lex.events if e.type in chain_types]
    assert emb_chain == chain_types
    assert lex_chain == chain_types


def test_unknown_signal_raises_configuration_error(instances):
    with pytest.raises(ConfigurationError, match="signal"):
        retrieve(instances[0], signal="bogus")


def test_hash_embedder_norm_is_one_or_zero():
    """L2-normalized vectors (or all-zero when no tokens). Defends the
    cosine-similarity interpretation downstream."""
    import math
    e = HashEmbedder()
    vs = e.embed(["the quick brown fox", "another sentence", ""])
    for v in vs:
        n = math.sqrt(sum(x * x for x in v))
        assert n == pytest.approx(1.0) or n == 0.0


def test_hash_embedder_deterministic_across_instances():
    e1 = HashEmbedder()
    e2 = HashEmbedder()
    assert e1.embed(["hello world"]) == e2.embed(["hello world"])


def test_openai_embedder_missing_key_raises_clean():
    """Production wiring's failure mode: clear ConfigurationError, not a
    cryptic ImportError or KeyError at first network call."""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            OpenAIEmbedder()
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


# ---------------------------------------------------------------------------
# Issue 2: embedding-error visibility + the tiktoken special-token root cause
# ---------------------------------------------------------------------------

MARKER = "<|endoftext|>"


class _FakeTurn:
    def __init__(self, turn_id, text):
        self.data = {"turn_id": turn_id, "text": text}


class _FakeView:
    def __init__(self, turns):
        self._turns = turns

    def objects(self, type):  # noqa: A002 — mirrors the real view API
        return list(self._turns) if type == "turn" else []


class _MarkerFailingEmbedder:
    """Simulates the tiktoken special-token ValueError: raises on any input
    containing MARKER, returns a unit vector otherwise."""

    model = "marker-failing"

    def embed(self, texts):
        out = []
        for t in texts:
            if MARKER in t:
                raise ValueError(f"disallowed special token in {t!r}")
            out.append([1.0, 0.0, 0.0, 0.0])
        return out


def test_score_embedding_isolates_and_reports_offending_turn():
    """A single pathological turn must NOT take down the whole question's
    scoring. The bad turn gets a neutral 0.0; good turns still score; the
    failure is reported with a turn_id + traceback + repr."""
    from regimes.agent.signals import score_embedding

    view = _FakeView([
        _FakeTurn("s1#0", "good turn one"),
        _FakeTurn("s1#1", f"poisoned {MARKER} turn"),
        _FakeTurn("s1#2", "good turn two"),
    ])
    errs = []
    scores = score_embedding(
        view, "good question", embedder=_MarkerFailingEmbedder(),
        on_error=errs.append,
    )
    # Good turns scored (unit·unit = 1.0); poisoned turn got 0.0.
    assert scores["s1#0"] == pytest.approx(1.0)
    assert scores["s1#2"] == pytest.approx(1.0)
    assert scores["s1#1"] == pytest.approx(0.0)
    # Exactly one auditable record, naming the offending turn, with detail.
    assert len(errs) == 1
    rec = errs[0]
    assert rec["turn_id"] == "s1#1"
    assert rec["exception_type"] == "ValueError"
    assert MARKER in rec["input_repr"]
    assert "Traceback" in rec["traceback"]
    assert rec["input_chars"] > 0


def test_score_embedding_clean_run_reports_no_errors():
    from regimes.agent.signals import score_embedding

    view = _FakeView([_FakeTurn("s1#0", "a turn"), _FakeTurn("s1#1", "another")])
    errs = []
    scores = score_embedding(
        view, "q", embedder=_MarkerFailingEmbedder(), on_error=errs.append,
    )
    assert errs == []
    assert set(scores) == {"s1#0", "s1#1"}


def test_embedding_error_surfaces_as_event_and_chain_continues(instances):
    """End-to-end: a poisoned question makes the embedder raise, but the
    agent emits an agent.embedding_error event (auditable, not silent) and
    the chain still reaches context.assembled with a neutral score."""
    from regimes.agent import events as AE
    from regimes.agent import reset_embedder

    try:
        set_embedder(_MarkerFailingEmbedder())
        t = retrieve(
            instances[0], signal="embedding", token_budget=2500,
            question=f"poisoned {MARKER} question",
        )
    finally:
        reset_embedder()

    err_events = [e for e in t.events if e.type == AE.EMBEDDING_ERROR]
    assert len(err_events) >= 1
    p = err_events[0].payload
    assert p["turn_id"] == "<question>"
    assert p["exception_type"] == "ValueError"
    assert MARKER in p["input_repr"]
    assert "Traceback" in p["traceback"]
    # The chain didn't die: assembly still ran, and the scored event
    # records the error count.
    assert any(e.type == "context.assembled" for e in t.events)
    scored = _scored_event(t.events)
    assert scored.payload["n_embedding_errors"] >= 1


def test_truncate_passes_disallowed_special_empty(monkeypatch):
    """Root-cause regression: _truncate_for_embedding must call tiktoken
    with disallowed_special=() so special-token substrings encode as
    ordinary text instead of raising ValueError."""
    from regimes.agent import embedders

    class _FakeEnc:
        def encode(self, text, *, disallowed_special="all"):
            # Mirror real tiktoken: default ("all") raises on special tokens.
            if disallowed_special != () and MARKER in text:
                raise ValueError("disallowed special token")
            return list(range(len(text)))

        def decode(self, toks):
            return "x" * len(toks)

    monkeypatch.setattr(embedders, "_TIKTOKEN_ENC", _FakeEnc())
    # Would raise ValueError under the old default; must NOT now.
    out = embedders._truncate_for_embedding(f"hello {MARKER} world")
    assert isinstance(out, str)
    assert embedders.embedding_token_count(f"a {MARKER} b") is not None


def test_custom_embedder_swap_is_honored(instances):
    """set_embedder() swaps the embedder; the scoring behavior sees it
    via get_embedder(). Proves the production wiring path."""

    class StubEmbedder:
        model = "stub-v0"

        def embed(self, texts):
            # constant unit vector → all turns score identically (1.0 cosine)
            return [[1.0] + [0.0] * 7 for _ in texts]

    try:
        set_embedder(StubEmbedder())
        t = retrieve(instances[0], signal="embedding", token_budget=2500)
        scored = _scored_event(t.events)
        assert scored.payload["scorer_model"] == "stub-v0"
        # every score should be 1.0 (cosine of identical unit vectors)
        scores = scored.payload["scores"]
        assert all(v == pytest.approx(1.0) for v in scores.values())
    finally:
        from regimes.agent import reset_embedder
        reset_embedder()
