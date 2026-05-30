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
            # Mimic a model that reads the requested signature out of the
            # prompt and emits a matching, gate-clean transform — so a
            # reader-prompt prompt yields a reader-prompt function, etc.
            prompt = messages[0]["content"]
            if "selected_turns" in prompt:
                return _Resp(
                    "```python\n"
                    "def transform(selected_turns, scores, question, question_date):\n"
                    "    return sorted(selected_turns, key=lambda t: -scores.get(t, 0.0))\n"
                    "```"
                )
            if "prompt_parts" in prompt:
                return _Resp(
                    "```python\n"
                    "def transform(prompt_parts, question, question_date):\n"
                    "    out = dict(prompt_parts)\n"
                    "    out['instruction'] = out.get('instruction', '') + \\\n"
                    "        ' Reconcile the evidence already in context before answering.'\n"
                    "    return out\n"
                    "```"
                )
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
# Per-type authoring: the prompt names the right signature + constraints
# for the type the action space requested, and a returned sample of each
# type passes that type's static gate. (No API call — the prompt is read
# off the recorded fake-client call; the static gate is pure.)
# ---------------------------------------------------------------------------


def _budget_failure() -> Outcome:
    return Outcome(
        question_id="q1", question_type="multi-session",
        is_abstention=False, answer_session_ids=("sG",),
        correct=False, truncated=True,
        scores={"sG#0": 0.6, "sN#0": 0.9},
        ranked=("sN#0", "sG#0"),
        selected_turn_ids=("sN#0",),
        decisions=({"turn_id": "sG#0", "included": False, "reason": "budget"},),
        gold_evidence_turn_ids=("sG#0",),
    )


def _reconciliation_failure() -> Outcome:
    # Evidence WAS selected (present in context) but the answer is wrong.
    return Outcome(
        question_id="qR", question_type="multi-session",
        is_abstention=False, answer_session_ids=("sG",),
        correct=False, judge_label="wrong",
        hypothesis="It was on a Tuesday.",
        scores={"sG#0": 0.9, "sN#0": 0.5},
        ranked=("sG#0", "sN#0"),
        selected_turn_ids=("sG#0", "sN#0"),
        gold_evidence_turn_ids=("sG#0",),
    )


def _static_gate_for(active_type):
    from regimes.targets.longmemeval.action_space import LongMemEvalActionSpace
    aspace = LongMemEvalActionSpace()
    aspace._active_type = active_type
    return aspace.static_gate


def test_score_transform_prompt_and_gate(_fake_anthropic):
    from regimes.targets.longmemeval.transform_types import SCORE_TRANSFORM

    author = build_real_author()
    cli = author._ensure_client()
    drafted = author.draft_typed(
        dominant_regime="budget-truncation",
        failures=[_budget_failure()],
        transform_type="score-transform",
    )
    prompt = cli.messages.calls[0]["messages"][0]["content"]
    # Right signature + score-transform constraints.
    assert "def transform(scores: dict, graph" in prompt
    assert "-> dict" in prompt
    assert "REWEIGHT" in prompt
    assert "SAME turn_ids" in prompt
    # Failure signals for the regime.
    assert "evidence_dropped_at_budget" in prompt
    assert "budget_winners" in prompt
    assert "sG#0" in prompt
    # A returned sample passes the score-transform static gate.
    r = _static_gate_for(SCORE_TRANSFORM)(drafted.source)
    assert r.passed, r.reasons


def test_assembly_transform_prompt_and_gate(_fake_anthropic):
    from regimes.targets.longmemeval.transform_types import ASSEMBLY_TRANSFORM

    author = build_real_author()
    cli = author._ensure_client()
    drafted = author.draft_typed(
        dominant_regime="assembly-crowding",
        failures=[_budget_failure()],
        transform_type="assembly-transform",
    )
    prompt = cli.messages.calls[0]["messages"][0]["content"]
    # Right signature: takes selected_turns + scores, returns a list.
    assert "def transform(selected_turns: list, scores: dict" in prompt
    assert "-> list" in prompt
    # Constraints: subset-or-reorder, no fabricated ids.
    assert "SUBSET-OR-REORDER" in prompt
    assert "fabricated" in prompt
    # Failure signals: dropped evidence + crowding competitors.
    assert "evidence_dropped_at_budget" in prompt
    assert "sG#0" in prompt
    # A returned sample passes the assembly-transform static gate.
    r = _static_gate_for(ASSEMBLY_TRANSFORM)(drafted.source)
    assert r.passed, r.reasons


def test_reader_prompt_transform_prompt_and_gate(_fake_anthropic):
    from regimes.targets.longmemeval.transform_types import READER_PROMPT_TRANSFORM

    author = build_real_author()
    cli = author._ensure_client()
    drafted = author.draft_typed(
        dominant_regime="assemble-internal",
        failures=[_reconciliation_failure()],
        transform_type="reader-prompt-transform",
    )
    prompt = cli.messages.calls[0]["messages"][0]["content"]
    # Right signature: edits prompt_parts, returns a dict.
    assert "def transform(prompt_parts: dict, question: str" in prompt
    assert "-> dict" in prompt
    # Constraints: same keys, bounded added text.
    assert "SAME KEYS" in prompt
    assert "2000" in prompt
    # Failure signals: reconciliation failures (evidence in context, wrong).
    assert "reconciliation_failure=True" in prompt
    assert "evidence_present_in_context" in prompt
    assert "sG#0" in prompt
    # A returned sample passes the reader-prompt static gate.
    r = _static_gate_for(READER_PROMPT_TRANSFORM)(drafted.source)
    assert r.passed, r.reasons


def test_action_space_routes_active_type_to_llm_author(_fake_anthropic):
    """End-to-end wiring: the action space selects the type for the
    regime, sets _active_type, and the LLMAuthor drafts THAT type — the
    drafted source then passes the action space's own static gate."""
    from regimes.targets.longmemeval.action_space import LongMemEvalActionSpace

    author = build_real_author()
    aspace = LongMemEvalActionSpace(author=author)
    # assemble-internal routes to reader-prompt-transform.
    change = aspace.draft(
        dominant_regime="assemble-internal",
        failures=[_reconciliation_failure()],
    )
    assert aspace._active_type.name == "reader-prompt-transform"
    assert "prompt_parts" in change.source
    r = aspace.static_gate(change.source)
    assert r.passed, r.reasons


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
