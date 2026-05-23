"""Tests for the LLMAuthor wiring on the --mode real path.

These exercise the construction path (build_real_author honors
BEHAVIORDRAFTS_MODEL, validates env), and the source-persistence
contract (transform_log entries carry the actual code the author
produced, not just the name + status). No real LLM call is made —
the Anthropic SDK is import-stubbed and the HTTP path is bypassed
via a fake client / a custom author that mimics the LLMAuthor
return shape.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Iterable

import pytest

from regimes.agent import transforms as T
from regimes.eval.types import Outcome
from regimes.loop import MockEval, MockInstance, run_loop
from regimes.loop.hypothesize import (
    DEFAULT_LLM_MODEL,
    DraftedTransform,
    LLMAuthor,
    build_real_author,
)


@pytest.fixture(autouse=True)
def _clean_pipeline():
    T.clear()
    yield
    T.clear()


@pytest.fixture
def _fake_anthropic(monkeypatch):
    """Inject a stand-in `anthropic` module so LLMAuthor construction
    succeeds without the real SDK installed in this container.

    The fake exposes the minimum surface LLMAuthor touches at
    construction time (just the module presence) and at draft() time
    (the `Anthropic` class with a `messages.create` that returns a
    canned text block)."""
    fake = types.ModuleType("anthropic")

    class _Block:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class _Resp:
        def __init__(self, text):
            self.content = [_Block(text)]

    class _Messages:
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, *, model, max_tokens, temperature, messages):
            self.calls.append({
                "model": model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature,
            })
            return _Resp(
                "```python\n"
                "def transform(scores, graph, question, question_date):\n"
                "    return {k: v * 1.0 for k, v in scores.items()}\n"
                "```"
            )

    class _Anthropic:
        def __init__(self, *, api_key):
            self.api_key = api_key
            self.messages = _Messages()

    fake.Anthropic = _Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    return fake


def test_build_real_author_defaults_to_claude_model(_fake_anthropic, monkeypatch):
    monkeypatch.delenv("BEHAVIORDRAFTS_MODEL", raising=False)
    author = build_real_author()
    assert isinstance(author, LLMAuthor)
    assert author.name == DEFAULT_LLM_MODEL


def test_build_real_author_honors_env_override(_fake_anthropic, monkeypatch):
    monkeypatch.setenv("BEHAVIORDRAFTS_MODEL", "claude-something-else")
    author = build_real_author()
    assert isinstance(author, LLMAuthor)
    assert author.name == "claude-something-else"


def test_llm_author_draft_invokes_client_and_returns_drafted(_fake_anthropic):
    author = build_real_author()
    # Force client now so we can inspect the recorded call list.
    cli = author._ensure_client()
    failures = [
        Outcome(
            question_id="q1", question_type="multi-session",
            is_abstention=False, answer_session_ids=("sG",),
            correct=False, truncated=True,
            scores={"sG#0": 0.6, "sN#0": 0.9},
            ranked=("sN#0", "sG#0"),
            selected_turn_ids=("sN#0",),
            decisions=(
                {"turn_id": "sG#0", "included": False, "reason": "budget"},
            ),
            gold_evidence_turn_ids=("sG#0",),
        ),
    ]
    drafted = author.draft(dominant_regime="budget-truncation",
                           failures=failures)
    assert isinstance(drafted, DraftedTransform)
    assert drafted.author == DEFAULT_LLM_MODEL
    assert "def transform" in drafted.source
    # The author received the per-question budget signals in its prompt.
    assert len(cli.messages.calls) == 1
    prompt = cli.messages.calls[0]["messages"][0]["content"]
    assert "budget-truncation" in prompt
    assert "evidence_dropped_at_budget" in prompt
    assert "sG#0" in prompt   # the dropped evidence turn id is named
    assert "budget_winners" in prompt
    assert "REWEIGHT" in prompt


# ---------------------------------------------------------------------------
# End-to-end: the loop runs with an LLMAuthor-like author and the
# transform_log records the authored source.
# ---------------------------------------------------------------------------


@dataclass
class _RecordingAuthor:
    """An author with the LLMAuthor shape that returns a fixed source.
    Lets us prove draft() is invoked AND that the source it returns
    ends up in transform_log without depending on a network round-trip."""

    name: str = "recording-author"
    source: str = (
        "def transform(scores, graph, question, question_date):\n"
        "    return dict(scores)\n"
    )
    calls: list[dict] = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def draft(self, *, dominant_regime: str,
              failures: Iterable[Outcome]) -> DraftedTransform:
        sample = list(failures)
        self.calls.append({
            "dominant_regime": dominant_regime, "n_failures": len(sample),
        })
        return DraftedTransform(
            name=f"recorded_{dominant_regime.replace('-', '_')}",
            source=self.source,
            target_regime=dominant_regime,
            author=self.name,
            rationale="test",
        )


def _budget_truncation_mix() -> list[MockInstance]:
    return [
        MockInstance("q_ok", "multi-session", False, ("s_ok",), True,
                     scores={"s_ok#0": 1.0}, selected_turn_ids=("s_ok#0",)),
        MockInstance(
            "q_bt1", "multi-session", False, ("sG1",), False,
            scores={"sG1#0": 0.7, "sN#0": 0.6},
            ranked=("sG1#0", "sN#0"),
            selected_turn_ids=("sN#0",), truncated=True,
            decisions=({"turn_id": "sG1#0", "included": False, "reason": "budget"},),
            candidate_turn_ids=("sG1#0", "sN#0"),
            gold_score_threshold=1.0,
        ),
    ]


def test_transform_log_persists_authored_source():
    author = _RecordingAuthor()
    rep = run_loop(
        eval_backend=MockEval(),
        instances=_budget_truncation_mix(),
        author=author,
        max_consecutive_discards=1,
    )
    # draft() actually fired.
    assert len(author.calls) >= 1
    assert author.calls[0]["dominant_regime"] == "budget-truncation"
    # The authored source is persisted in every transform_log entry
    # (status varies by which gate the transform reached, but the code
    # itself must be there for the audit log to be reconstructable).
    assert rep.transform_log, "expected at least one transform_log entry"
    for rec in rep.transform_log:
        assert rec.get("source", "").strip(), \
            f"transform_log entry missing source: {rec}"
        assert "def transform" in rec["source"]
        assert rec.get("author") == "recording-author"


def test_run_loop_script_real_branch_constructs_llm_author(
    _fake_anthropic, monkeypatch
):
    """The construction path: --mode real builds an LLMAuthor via
    build_real_author and passes it through to run_loop. We don't exec
    the script end-to-end (RealEval requires keys, an LME checkout, and
    a dataset path); we just confirm:
      1) `scripts/run_loop.py` imports `build_real_author` from
         `regimes.loop.hypothesize` (not a fallback to StubAuthor), and
      2) the same factory yields an LLMAuthor under the same env, AND
      3) the source it calls names `build_real_author(` on the real
         branch.
    """
    import importlib.util
    from pathlib import Path

    monkeypatch.delenv("BEHAVIORDRAFTS_MODEL", raising=False)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_loop.py"
    spec = importlib.util.spec_from_file_location("run_loop_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    # Same factory the script will call on the real branch.
    assert mod.build_real_author is build_real_author
    author = mod.build_real_author()
    assert isinstance(author, LLMAuthor)
    # And the script does construct it inside the real branch (and
    # passes it to run_loop) — guard against a regression that drops
    # it and lets the runner fall back to StubAuthor.
    src = script_path.read_text()
    assert "build_real_author()" in src
    assert "author=author" in src
