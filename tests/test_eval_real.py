"""Unit tests for the RealEval wrapper against the synthetic fixture.

These tests use FakeReader + FakeJudge so they need no API keys, no LME
checkout, and no network. They prove:
  - the wrapper composes agent + reader + judge correctly
  - hypotheses.jsonl + references.json + aggregate.json are written in
    the expected shapes
  - Outcome records carry every field diagnose needs (selected_turn_ids,
    ranked, scores, decisions, answer_session_ids, ...)
  - per-type aggregation + accuracy math is correct
  - LMEJudge / AnthropicReader fail cleanly with ConfigurationError
    when their preconditions are missing
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from activegraph import ConfigurationError

from regimes.eval import (
    AnthropicReader,
    EvalResult,
    FakeJudge,
    FakeReader,
    LMEJudge,
    Outcome,
    RealEval,
)
from regimes.split import load_split

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "synthetic_lme.json"
SPLIT = REPO / "config" / "split.json"


@pytest.fixture(scope="module")
def all_instances():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def optimize_instances(all_instances):
    # NOTE: config/split.json now points at the real LME corpus (opaque
    # IDs not in the synthetic fixture), so we can't drive RealEval
    # tests through load_split anymore. Sample the synthetic fixture
    # directly — the tests below validate the eval WRAPPER, not the
    # split mechanics; split has its own test module.
    return all_instances[:24]


def _make_eval(signal: str = "embedding") -> RealEval:
    return RealEval(reader=FakeReader(), judge=FakeJudge(), signal=signal, token_budget=2500)


# ---------------------------------------------------------------------------
# RealEval wiring
# ---------------------------------------------------------------------------

def test_real_eval_runs_end_to_end(optimize_instances, tmp_path):
    ev = _make_eval()
    res = ev.run_on_split(optimize_instances, run_dir=tmp_path / "run")
    assert isinstance(res, EvalResult)
    assert res.backend == "real"
    assert len(res.outcomes) == len(optimize_instances)
    assert all(isinstance(o, Outcome) for o in res.outcomes)


def test_artifacts_written(optimize_instances, tmp_path):
    run = tmp_path / "run"
    ev = _make_eval()
    res = ev.run_on_split(optimize_instances, run_dir=run)
    assert (run / "hypotheses.jsonl").exists()
    assert (run / "references.json").exists()
    assert (run / "aggregate.json").exists()
    # hypotheses.jsonl is one record per line
    lines = [json.loads(l) for l in (run / "hypotheses.jsonl").read_text().splitlines() if l]
    assert len(lines) == len(optimize_instances)
    assert {"question_id", "hypothesis"} <= set(lines[0].keys())
    # aggregate written matches the in-memory result
    agg = json.loads((run / "aggregate.json").read_text())
    assert agg["overall_accuracy"] == res.aggregate["overall_accuracy"]


# ---------------------------------------------------------------------------
# Outcome shape — diagnose needs each field
# ---------------------------------------------------------------------------

def test_outcome_carries_full_audit(optimize_instances, tmp_path):
    ev = _make_eval()
    res = ev.run_on_split(optimize_instances[:3], run_dir=tmp_path / "run")
    o = res.outcomes[0]
    # identity
    assert o.question_id
    assert o.question_type in {
        "single-session-user", "single-session-assistant",
        "single-session-preference", "multi-session",
        "temporal-reasoning", "knowledge-update",
    }
    assert isinstance(o.answer_session_ids, tuple)
    # agent output
    assert o.signal == "embedding"
    assert isinstance(o.selected_turn_ids, tuple)
    assert isinstance(o.ranked, tuple)
    assert isinstance(o.scores, dict)
    assert isinstance(o.decisions, tuple)
    assert isinstance(o.applied_transforms, tuple)
    # judge
    assert isinstance(o.correct, bool)
    assert o.judge_label  # FakeJudge always assigns a label
    # run linkage
    assert o.run_id


def test_aggregate_math_consistent(optimize_instances, tmp_path):
    res = _make_eval().run_on_split(optimize_instances, run_dir=tmp_path / "run")
    overall = res.aggregate["overall_accuracy"]
    manual = sum(1 for o in res.outcomes if o.correct) / len(res.outcomes)
    assert overall == manual
    # per-type
    per_type = res.aggregate["per_type_accuracy"]
    pt_seen = {o.question_type for o in res.outcomes}
    assert set(per_type.keys()) == pt_seen


def test_per_type_helper_matches_aggregate(optimize_instances, tmp_path):
    res = _make_eval().run_on_split(optimize_instances, run_dir=tmp_path / "run")
    assert res.per_type_accuracy() == res.aggregate["per_type_accuracy"]


# ---------------------------------------------------------------------------
# Signal flow proof: rag-dense comparison uses embedding
# ---------------------------------------------------------------------------

def test_signal_propagates_into_outcomes(optimize_instances, tmp_path):
    for sig in ("lexical", "embedding"):
        res = RealEval(reader=FakeReader(), judge=FakeJudge(), signal=sig,
                       token_budget=2500).run_on_split(
            optimize_instances[:3], run_dir=tmp_path / f"run_{sig}",
        )
        assert all(o.signal == sig for o in res.outcomes)
        assert res.config["signal"] == sig


# ---------------------------------------------------------------------------
# Helper functions on Outcome (regime-detection inputs)
# ---------------------------------------------------------------------------

def test_outcome_gold_helpers():
    # Construct a hand-rolled Outcome that exercises the helpers directly
    o = Outcome(
        question_id="multi_session_q005",
        question_type="multi-session",
        is_abstention=False,
        answer_session_ids=("multi_session_q005_sess2", "multi_session_q005_sess4"),
        correct=False,
        selected_turn_ids=(
            "multi_session_q005_sess0#1",     # not gold
            "multi_session_q005_sess2#3",     # gold
        ),
        ranked=(
            "multi_session_q005_sess1#0",     # not gold
            "multi_session_q005_sess4#2",     # gold (top-2)
            "multi_session_q005_sess0#1",     # not gold
            "multi_session_q005_sess2#3",     # gold
        ),
        scores={
            "multi_session_q005_sess0#1": 3.0,
            "multi_session_q005_sess1#0": 4.0,
            "multi_session_q005_sess2#3": 1.5,
            "multi_session_q005_sess4#2": 3.5,
        },
        truncated=True,
    )
    assert o.gold_selected() == ("multi_session_q005_sess2#3",)
    assert o.gold_ranked_top_k(2) == ("multi_session_q005_sess4#2",)
    assert o.gold_ranked_top_k(4) == (
        "multi_session_q005_sess4#2",
        "multi_session_q005_sess2#3",
    )
    assert o.gold_max_score() == 3.5


# ---------------------------------------------------------------------------
# Production-path failure modes (no keys here)
# ---------------------------------------------------------------------------

def test_lme_judge_missing_checkout_raises():
    with pytest.raises(ConfigurationError, match="does not exist"):
        LMEJudge(lme_checkout="/no/such/path/__lme__")


def test_lme_judge_missing_openai_key_raises(tmp_path):
    """Simulate a 'partially valid' LME checkout: dir exists + submodule
    file exists, but no OPENAI_API_KEY. Must still raise."""
    root = tmp_path / "lme"
    (root / "third_party/longmemeval/src/evaluation").mkdir(parents=True)
    (root / "third_party/longmemeval/src/evaluation/evaluate_qa.py").write_text("# stub")
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            LMEJudge(lme_checkout=str(root))
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def test_anthropic_reader_missing_key_raises():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            AnthropicReader()
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


# ---------------------------------------------------------------------------
# FakeJudge consistency (test of test infra)
# ---------------------------------------------------------------------------

def test_fake_judge_scores_via_gold_overlap_rule(optimize_instances, tmp_path):
    """FakeJudge says correct iff any selected turn's session is in gold.
    Validate that rule on real wrapper output so future Outcome additions
    don't silently break the test infra."""
    res = _make_eval().run_on_split(optimize_instances[:5], run_dir=tmp_path / "run")
    for o in res.outcomes:
        if o.is_abstention:
            expected = (len(o.selected_turn_ids) == 0)
        else:
            sids = {tid.split("#", 1)[0] for tid in o.selected_turn_ids if "#" in tid}
            expected = bool(sids & set(o.answer_session_ids))
        assert o.correct == expected
