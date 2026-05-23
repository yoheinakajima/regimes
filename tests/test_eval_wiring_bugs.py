"""Regression guards for two eval-wiring bugs found on the real path.

Bug A: the loop path produced empty hypotheses for all 50 OPTIMIZE
       questions while the standalone path produced full answers. The
       silent failure mode was: agent.retrieve returned empty context
       and the reader returned "" — but ALL 50 being empty looked like
       a one-off rather than a systemic break, because nothing in the
       pipeline asserts non-empty output.

Bug B: LMEJudge subprocess was handed relative paths and crashed with
       FileNotFoundError when resolving them against the subprocess's
       cwd (the LME checkout, not the regimes repo).

These tests pin BOTH guarantees without API keys or network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from regimes.eval import LMEJudge, RealEval
from regimes.eval.types import Outcome
from regimes.loop import run_loop
from regimes.split import load_split

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "synthetic_lme.json"
SPLIT = REPO / "config" / "split.json"


# ===========================================================================
# Bug A regression guard
# ===========================================================================


@dataclass
class ContextEchoReader:
    """A reader that ECHOES the context length as its hypothesis.

    This is the canary for Bug A: if the agent's chain doesn't fire
    inside the loop's runtime and assembled context is empty, the
    hypothesis will be "context_len=0" — easy to assert on.

    A reader that just returned a fixed string would have masked the
    bug; the original FakeReader took a context-head substring which
    would have looked plausible. This one is built so "the agent did
    not produce real context" cannot pass silently.
    """

    name: str = "context-echo"
    seen: list[tuple[str, int]] = field(default_factory=list)

    def answer(self, *, context: str, question: str, question_id: str) -> str:
        self.seen.append((question_id, len(context)))
        return f"context_len={len(context)} qid={question_id}"


def _instances() -> list[dict]:
    s = load_split(SPLIT)
    all_insts = json.loads(FIXTURE.read_text())
    by_id = {x["question_id"]: x for x in all_insts}
    # 5 is enough to demonstrate the property; the original bug hit all 50.
    return [by_id[q] for q in s.optimize[:5]]


@dataclass
class _StubJudge:
    """A judge that records what it was called with and returns
    deterministic verdicts; lets us assert reader behaviour without
    needing LME's subprocess."""

    name: str = "stub-judge"
    last_hypotheses: list[dict] | None = None

    def judge(self, *, hypotheses_path, references_path, run_dir):
        self.last_hypotheses = [
            json.loads(l) for l in Path(hypotheses_path).read_text().splitlines() if l
        ]
        refs = json.loads(Path(references_path).read_text())
        return [{"question_id": r["question_id"], "correct": False, "label": "0"}
                for r in refs]


def test_loop_path_produces_non_empty_hypotheses():
    """Standalone vs loop must not diverge on hypothesis emptiness.

    This is the guard for Bug A: if agent.retrieve produces empty
    context inside the loop's runtime (e.g. because the behavior
    registry got clobbered or the agent's Runtime can't see its own
    behaviors), the reader will see len(context)==0 and we fail loud.
    """
    reader = ContextEchoReader()
    judge = _StubJudge()
    backend = RealEval(reader=reader, judge=judge,
                       signal="embedding", token_budget=2500)

    class _RD:
        """The same wrapper scripts/run_loop.py uses; reproduces the
        invocation context that originally exposed Bug A."""
        def __init__(self, ev, base):
            self.ev, self.base, self.n = ev, Path(base), 0
        def run_on_split(self, insts):
            self.n += 1
            return self.ev.run_on_split(insts, run_dir=self.base / f"sub_{self.n}")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        wrapped = _RD(backend, td)
        rep = run_loop(eval_backend=wrapped, instances=_instances(),
                       pause_after="histogram")

    # The reader must have been called once per instance.
    assert len(reader.seen) == len(_instances()), (
        f"reader called {len(reader.seen)} times; expected {len(_instances())}"
    )

    # EVERY context must be non-empty. This is the property Bug A broke.
    empties = [(qid, n) for qid, n in reader.seen if n == 0]
    assert not empties, (
        f"agent.retrieve produced empty context for {len(empties)} of "
        f"{len(reader.seen)} questions: {empties[:5]}"
    )

    # And the eval-emitted hypotheses must also be non-empty.
    baseline = rep.baseline
    assert baseline is not None
    # Per-question outcome summaries don't carry hypothesis directly,
    # so check the stub judge's record.
    assert judge.last_hypotheses is not None
    empties_jsonl = [h for h in judge.last_hypotheses if not h["hypothesis"]]
    assert not empties_jsonl, (
        f"{len(empties_jsonl)} of {len(judge.last_hypotheses)} hypotheses "
        f"written to disk are empty — Bug A regression."
    )


def test_loop_path_matches_standalone_path_on_context_lengths():
    """Stronger guard: the per-question context length the reader sees
    in the loop path must equal what the standalone path produces. If
    something about the loop's invocation context (registry state,
    nested runtime) shrinks context relative to standalone, this fails.
    """
    insts = _instances()

    # --- standalone ---
    standalone_reader = ContextEchoReader()
    standalone = RealEval(reader=standalone_reader, judge=_StubJudge(),
                          signal="embedding", token_budget=2500)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        standalone.run_on_split(insts, run_dir=Path(td) / "standalone")
    standalone_seen = dict(standalone_reader.seen)

    # --- loop ---
    loop_reader = ContextEchoReader()
    loop_backend = RealEval(reader=loop_reader, judge=_StubJudge(),
                            signal="embedding", token_budget=2500)

    class _RD:
        def __init__(self, ev, base): self.ev, self.base, self.n = ev, Path(base), 0
        def run_on_split(self, insts):
            self.n += 1
            return self.ev.run_on_split(insts, run_dir=self.base / f"sub_{self.n}")

    with tempfile.TemporaryDirectory() as td:
        wrapped = _RD(loop_backend, td)
        run_loop(eval_backend=wrapped, instances=insts, pause_after="histogram")
    loop_seen = dict(loop_reader.seen)

    assert standalone_seen.keys() == loop_seen.keys()
    diffs = {qid: (standalone_seen[qid], loop_seen[qid])
             for qid in standalone_seen
             if standalone_seen[qid] != loop_seen[qid]}
    assert not diffs, (
        f"loop path diverged from standalone on context lengths: {diffs}"
    )


def test_realeval_surfaces_empty_context_as_outcome_error():
    """Defense-in-depth: even if a future bug DOES produce empty
    context, RealEval must mark the outcome's `error` field so it can
    never look like a legitimate reader response."""

    # Build a stripped instance with zero turns so the agent's chain
    # produces an empty context legitimately. The error must surface.
    insts = [{
        "question_id": "q_zero",
        "question": "anything?",
        "question_date": "2024-01-01",
        "answer": "x",
        "question_type": "multi-session",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": [],
        "haystack_dates": [],
        "haystack_sessions": [],
    }]
    reader = ContextEchoReader()
    judge = _StubJudge()
    backend = RealEval(reader=reader, judge=judge,
                       signal="embedding", token_budget=2500)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = backend.run_on_split(insts, run_dir=Path(td) / "x")
    o = res.outcomes[0]
    # Either the agent raised on the empty corpus, or it returned empty
    # context — EITHER WAY, .error must be populated, never silently empty.
    assert o.error, (
        "RealEval allowed an empty-context outcome to slip through "
        "with .error==None — Bug A could regress unnoticed."
    )


# ===========================================================================
# Bug B regression guard
# ===========================================================================


def test_lme_judge_uses_absolute_paths(tmp_path, monkeypatch):
    """LMEJudge.judge must pass ABSOLUTE paths to the subprocess.

    The subprocess runs with cwd=<LME checkout>; relative paths resolve
    against that, not the regimes repo. The original bug was that the
    judge was handed `runs/loop_001/sub_1/hypotheses.jsonl` and the
    subprocess couldn't find the file even though it existed under the
    regimes repo.

    We monkeypatch subprocess.run to capture the argv and assert each
    path arg is absolute.
    """
    # Build a fake LME checkout layout so LMEJudge.__post_init__ accepts.
    fake_lme = tmp_path / "fake_lme"
    eval_py = fake_lme / "third_party/longmemeval/src/evaluation/evaluate_qa.py"
    eval_py.parent.mkdir(parents=True)
    eval_py.write_text("# stub\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")

    judge = LMEJudge(lme_checkout=str(fake_lme))

    # Write hypotheses + references files using RELATIVE paths from the
    # current working directory — the original bug's invocation shape.
    rel_dir = tmp_path / "runs" / "loop_001" / "sub_1"
    rel_dir.mkdir(parents=True)
    hyp_rel = rel_dir / "hypotheses.jsonl"
    hyp_rel.write_text(json.dumps({"question_id": "q1", "hypothesis": "y"}) + "\n")
    ref_rel = rel_dir / "references.json"
    ref_rel.write_text(json.dumps([{"question_id": "q1", "question": "q",
                                    "answer": "a", "question_type": "multi-session",
                                    "answer_session_ids": [],
                                    "is_abstention": False}]) + "\n")

    # Also write a stub results file the judge will try to parse so the
    # subprocess "success" path doesn't error on missing output.
    results_path = hyp_rel.with_name(hyp_rel.name + f".eval-results-{judge.name}")
    results_path.write_text(
        json.dumps({"question_id": "q1", "autoeval_label": "1"}) + "\n"
    )

    # Use paths that LOOK relative (string form, no leading slash) to
    # mimic the buggy caller. They live under tmp_path so they exist.
    rel_hyp_str = os.path.relpath(hyp_rel)
    rel_ref_str = os.path.relpath(ref_rel)
    rel_run_str = os.path.relpath(rel_dir)
    assert not Path(rel_hyp_str).is_absolute()
    assert not Path(rel_ref_str).is_absolute()

    captured: dict = {}

    def fake_run(cmd, *a, **kw):
        # Record argv. We require that every path-looking arg the
        # subprocess receives is absolute, because cwd inside is the
        # LME checkout — relative paths would resolve there.
        captured["cmd"] = list(cmd)
        captured["cwd"] = kw.get("cwd")

        class _Res:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Res()

    monkeypatch.setattr(subprocess, "run", fake_run)
    judge.judge(
        hypotheses_path=rel_hyp_str,
        references_path=rel_ref_str,
        run_dir=rel_run_str,
    )

    # Path args are positions [2], [3] (hypotheses, references) in
    # [python, eval_py, judge_name, hyp, ref]. judge_name (slot [2]) is
    # not a path; the model name; slots [1], [3], [4] are paths.
    assert len(captured["cmd"]) == 5, captured["cmd"]
    for i in (1, 3, 4):
        p = captured["cmd"][i]
        assert Path(p).is_absolute(), (
            f"arg {i} ({p!r}) is not absolute — Bug B regression "
            f"(subprocess cwd is {captured['cwd']}, "
            f"relative path would not resolve there)"
        )
    # The subprocess cwd is the LME root and must also be absolute.
    assert Path(captured["cwd"]).is_absolute()


def test_lme_judge_captures_stderr_into_log(tmp_path, monkeypatch):
    """When the subprocess fails, both stdout and stderr must end up in
    eval.log so the failure is debuggable without re-running."""
    fake_lme = tmp_path / "fake_lme"
    (fake_lme / "third_party/longmemeval/src/evaluation").mkdir(parents=True)
    (fake_lme / "third_party/longmemeval/src/evaluation/evaluate_qa.py").write_text("# stub")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    judge = LMEJudge(lme_checkout=str(fake_lme))

    rel_dir = tmp_path / "out"
    rel_dir.mkdir()
    hyp = rel_dir / "h.jsonl"
    hyp.write_text("{}\n")
    ref = rel_dir / "r.json"
    ref.write_text("[]\n")

    def failing_run(cmd, *a, **kw):
        raise subprocess.CalledProcessError(
            returncode=2, cmd=cmd, output="some stdout",
            stderr="FileNotFoundError: runs/loop_001/sub_1/hypotheses.jsonl",
        )
    monkeypatch.setattr(subprocess, "run", failing_run)

    with pytest.raises(RuntimeError) as exc:
        judge.judge(
            hypotheses_path=str(hyp),
            references_path=str(ref),
            run_dir=str(rel_dir),
        )
    log = (rel_dir / "eval.log").read_text()
    assert "FileNotFoundError" in log
    assert "stderr" in log
    # The raised RuntimeError must also surface the stderr tail so
    # the loop event log carries a useful message.
    assert "FileNotFoundError" in str(exc.value)
