"""MockEval — the in-memory backend the loop uses for keyless testing.

Pins the contract: MockInstance in, EvalResult/Outcome out, transform
pipeline honored, aggregate math sane.
"""

from __future__ import annotations

import pytest

from regimes.agent import transforms as T
from regimes.eval.types import EvalResult, Outcome
from regimes.loop.mock_eval import MockEval, MockInstance


@pytest.fixture(autouse=True)
def _clean_pipeline():
    T.clear()
    yield
    T.clear()


def test_mock_eval_runs_and_returns_outcomes():
    insts = [
        MockInstance("q1", "multi-session", False, ("s1",), True,
                     scores={"s1#0": 1.0}, selected_turn_ids=("s1#0",)),
        MockInstance("q2", "multi-session", False, ("s2",), False,
                     scores={"s2#0": 0.1}),
    ]
    res = MockEval().run_on_split(insts)
    assert isinstance(res, EvalResult)
    assert res.backend == "mock"
    assert len(res.outcomes) == 2
    assert {o.question_id for o in res.outcomes} == {"q1", "q2"}


def test_mock_eval_aggregate_math():
    insts = [
        MockInstance(f"q{i}", "multi-session", False, (f"s{i}",), i % 2 == 0,
                     scores={f"s{i}#0": 0.5})
        for i in range(4)
    ]
    res = MockEval().run_on_split(insts)
    assert res.aggregate["n"] == 4
    assert res.aggregate["overall_accuracy"] == 0.5
    assert res.aggregate["per_type_accuracy"]["multi-session"] == 0.5


def test_mock_eval_carries_score_error_through():
    insts = [
        MockInstance("q_err", "multi-session", False, ("s1",), False,
                     score_error="agent.score_embedding:BadRequestError: x"),
    ]
    res = MockEval().run_on_split(insts)
    assert res.outcomes[0].score_error.startswith("agent.score_embedding")
    assert res.aggregate["n_errors"] == 1


def test_mock_eval_applies_registered_transform():
    # Baseline incorrect; transform pushes gold score over threshold, candidate
    # is in selected_turn_ids' candidate set → flips to correct.
    def boost(scores, graph, q, qd):
        return {t: s * 2 for t, s in scores.items()}

    T.promote("boost2x", boost)
    insts = [
        MockInstance("q1", "multi-session", False, ("s1",), False,
                     scores={"s1#0": 0.6, "s2#0": 0.4},
                     candidate_turn_ids=("s1#0", "s2#0"),
                     gold_score_threshold=1.0),
    ]
    res = MockEval().run_on_split(insts)
    assert res.outcomes[0].correct is True
    assert res.outcomes[0].applied_transforms == ("boost2x",)


def test_mock_eval_passthrough_when_no_transform_registered():
    insts = [
        MockInstance("q1", "multi-session", False, ("s1",), False,
                     scores={"s1#0": 0.5}),
    ]
    res = MockEval().run_on_split(insts)
    assert res.outcomes[0].applied_transforms == ()
    assert res.outcomes[0].scores == {"s1#0": 0.5}
