"""Re-ingest equality + decision-snapshot stability.

Two properties, both required before any later number is trustworthy:

  1. Re-ingest equality: building the corpus graph twice on the same
     instance under FrozenClock + fresh IDGen produces a byte-identical
     event log. Mirrors LME's `re-ingest-equality` property test.

  2. Decision-snapshot stability: running the full retrieve() chain
     twice on the same question produces identical context.assembled
     payloads (selected_turn_ids, n_seeds, n_expanded, decisions, text).

Both run via the real activegraph package — `Graph(ids=IDGen(),
clock=FrozenClock(...))`, real `add_object`/`add_relation`, real
`Runtime` driving the behavior chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regimes.agent import ingest, retrieve

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "synthetic_lme.json"


def _serialize_event(e) -> dict:
    """Stable dict shape for byte-comparison. Drops nothing; the entire
    Event surface must match."""
    return {
        "id": e.id,
        "type": e.type,
        "payload": e.payload,
        "actor": e.actor,
        "frame_id": e.frame_id,
        "caused_by": e.caused_by,
        "timestamp": e.timestamp,
    }


@pytest.fixture(scope="module")
def instances() -> list[dict]:
    return json.loads(FIXTURE.read_text())


# ---------- 1) re-ingest equality ----------

def test_reingest_equality_first_instance(instances) -> None:
    """Ingest twice. Compare event logs as serialized JSON."""
    inst = instances[0]
    g1, _ = ingest(inst)
    g2, _ = ingest(inst)
    log1 = [_serialize_event(e) for e in g1.events]
    log2 = [_serialize_event(e) for e in g2.events]
    s1 = json.dumps(log1, sort_keys=True)
    s2 = json.dumps(log2, sort_keys=True)
    assert s1 == s2, "re-ingest event log differs"


def test_reingest_equality_three_distinct_instances(instances) -> None:
    """Cover three different question_types so the property covers
    different haystack shapes (not just one)."""
    sample = [instances[0], instances[33], instances[60]]
    for inst in sample:
        g1, _ = ingest(inst)
        g2, _ = ingest(inst)
        l1 = json.dumps([_serialize_event(e) for e in g1.events], sort_keys=True)
        l2 = json.dumps([_serialize_event(e) for e in g2.events], sort_keys=True)
        assert l1 == l2, f"re-ingest log diverged for {inst['question_id']}"


# ---------- 2) decision-snapshot stability across full retrieve ----------

def _decision_snapshot(trace) -> dict:
    """The pieces of the assembled context that must be identical
    across re-runs of the same question. Excludes nothing the loop
    cares about."""
    return {
        "selected_turn_ids": trace.context.meta.get("selected_turn_ids", []),
        "n_seeds": trace.context.meta.get("n_seeds"),
        "n_expanded": trace.context.meta.get("n_expanded"),
        "running_tokens": trace.context.meta.get("running_tokens"),
        "truncated": trace.context.truncated,
        "text": trace.context.text,
    }


def test_decision_snapshot_stable_first_instance(instances) -> None:
    inst = instances[0]
    t1 = retrieve(inst, token_budget=2500)
    t2 = retrieve(inst, token_budget=2500)
    assert _decision_snapshot(t1) == _decision_snapshot(t2)


def test_decision_snapshot_stable_across_types(instances) -> None:
    """Stability holds across different question-type shapes."""
    sample = [instances[0], instances[33], instances[60], instances[120]]
    for inst in sample:
        t1 = retrieve(inst, token_budget=2500)
        t2 = retrieve(inst, token_budget=2500)
        assert _decision_snapshot(t1) == _decision_snapshot(t2), (
            f"snapshot diverged for {inst['question_id']}"
        )


# ---------- 3) full event log is identical across re-run (strongest property) ----------

def test_full_event_log_byte_identical_across_retrieve(instances) -> None:
    """Strongest determinism claim: not just the assembled output but the
    entire runtime event log — including every behavior.started /
    behavior.completed lifecycle event from the runtime — is byte-identical
    across two runs of retrieve()."""
    inst = instances[0]
    t1 = retrieve(inst, token_budget=2500)
    t2 = retrieve(inst, token_budget=2500)
    l1 = [_serialize_event(e) for e in t1.events]
    l2 = [_serialize_event(e) for e in t2.events]
    assert json.dumps(l1, sort_keys=True) == json.dumps(l2, sort_keys=True)


# ---------- 4) package-usage proof: real event types present ----------

REQUIRED_PACKAGE_EVENT_TYPES = {
    "object.created",
    "relation.created",
    "behavior.started",
    "behavior.completed",
    "runtime.idle",
}

REQUIRED_AGENT_EVENT_TYPES = {
    "question.asked",
    "turns.scored",
    "turns.transformed",
    "turns.expanded",
    "context.assembled",
}


def test_event_log_contains_real_package_events(instances) -> None:
    """Proof that we're using the real runtime, not a lookalike. Both
    package-internal events (object.created, behavior.started, ...) and
    our custom agent events must appear."""
    trace = retrieve(instances[0], token_budget=2500)
    types_seen = {e.type for e in trace.events}
    missing_pkg = REQUIRED_PACKAGE_EVENT_TYPES - types_seen
    missing_agent = REQUIRED_AGENT_EVENT_TYPES - types_seen
    assert not missing_pkg, f"missing package events: {missing_pkg}"
    assert not missing_agent, f"missing agent events: {missing_agent}"


def test_event_causal_chain(instances) -> None:
    """The four custom events form a single causal chain rooted at
    question.asked. Proof the runtime fired our @behavior decorations
    in the expected order."""
    trace = retrieve(instances[0], token_budget=2500)
    by_id = {e.id: e for e in trace.events}
    chain = []
    for e in trace.events:
        if e.type in REQUIRED_AGENT_EVENT_TYPES:
            chain.append(e)
    assert [e.type for e in chain] == [
        "question.asked",
        "turns.scored",
        "turns.transformed",
        "turns.expanded",
        "context.assembled",
    ]
    # walk caused_by backwards from context.assembled to question.asked
    cur = chain[-1]
    visited = [cur.type]
    while cur.caused_by is not None:
        cur = by_id[cur.caused_by]
        visited.append(cur.type)
    assert visited == [
        "context.assembled",
        "turns.expanded",
        "turns.transformed",
        "turns.scored",
        "question.asked",
    ]
