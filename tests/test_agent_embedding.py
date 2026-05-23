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
