"""LME-harness wrapper.

Flow per `RealEval.run_on_split(instances, run_dir, signal, ...)`:

  1. For each instance:
       a) agent.retrieve(...) → AssembledContext + the full event log
       b) reader.answer(context, question) → hypothesis string
  2. Write `<run_dir>/hypotheses.jsonl` in LME's expected one-per-line
     {question_id, hypothesis} format.
  3. Write `<run_dir>/references.json` (the gold instance metadata, in
     LME's reference shape — answer, question_type, answer_session_ids).
  4. Call `judge.judge(hyp_path, ref_path, run_dir)` to produce the
     per-question verdict list. LMEJudge shells to LME's upstream
     evaluate_qa.py and reads the per-question result file.
  5. Build `Outcome` records joining traces + verdicts.
  6. Compute the EvalResult aggregate from the outcomes.

Construction-time validation:
  - LMEJudge requires `lme_checkout` to exist, third_party/longmemeval
    submodule to be initialized, and OPENAI_API_KEY in env.
  - AnthropicReader requires ANTHROPIC_API_KEY in env + `anthropic` pkg.
  - Both raise activegraph.ConfigurationError (caller-fixable) on any
    of those failing — per the framework's failure model.

For unit tests (no keys, no LME), use FakeReader + FakeJudge below.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from activegraph import ConfigurationError

from regimes.agent import retrieve as agent_retrieve
from regimes.eval.types import EvalResult, Judge, Outcome, Reader


REGIMES_RUN_VERSION = "regimes-eval-real-v1"


# ============================================================================
# Reader implementations
# ============================================================================


@dataclass
class AnthropicReader:
    """Real reader: Claude (tool-free, T=0, no web). Wired only to make
    requests work; not exercised in this container's tests."""

    name: str = "claude-sonnet-4-5"
    temperature: float = 0.0
    max_tokens: int = 1024
    _client: Any = None

    def __post_init__(self) -> None:
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ConfigurationError(
                "AnthropicReader requires ANTHROPIC_API_KEY in the environment."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ConfigurationError(
                "AnthropicReader requires the `anthropic` package. "
                "Install: pip install anthropic"
            ) from e

    def _ensure_client(self):  # pragma: no cover — network path
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def answer(self, *, context: str, question: str, question_id: str) -> str:  # pragma: no cover
        cli = self._ensure_client()
        prompt = (
            "Use only the provided context to answer the question. "
            "If the context does not contain the answer, say so explicitly.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        )
        resp = cli.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Take first text block
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""


@dataclass
class FakeReader:
    """Deterministic test reader. Returns a fingerprint of the context
    so unit tests can pin expected verdicts."""

    name: str = "fake-reader-v1"

    def answer(self, *, context: str, question: str, question_id: str) -> str:
        # Hypothesis = a marker the FakeJudge knows how to score.
        head = context[:120].replace("\n", " ")
        return f"[fake-reader hypothesis for {question_id}] {head}"


# ============================================================================
# Judge implementations
# ============================================================================


@dataclass
class LMEJudge:
    """Shells out to LME's pinned upstream judge (gpt-4o-2024-08-06).

    Reads per-question results from `<hyp>.eval-results-<judge>` and
    aggregate metrics via LME's print_qa_metrics.py wrapper.
    """

    lme_checkout: str
    name: str = "gpt-4o"

    def __post_init__(self) -> None:
        root = Path(self.lme_checkout)
        if not root.exists():
            raise ConfigurationError(
                f"LMEJudge: lme_checkout does not exist: {root}"
            )
        eval_py = root / "third_party/longmemeval/src/evaluation/evaluate_qa.py"
        if not eval_py.exists():
            raise ConfigurationError(
                f"LMEJudge: upstream judge missing at {eval_py}. "
                "Initialize the LME submodule (`git submodule update --init`) "
                "and re-run."
            )
        if "OPENAI_API_KEY" not in os.environ:
            raise ConfigurationError(
                "LMEJudge requires OPENAI_API_KEY in the environment."
            )

    def judge(
        self,
        *,
        hypotheses_path: str,
        references_path: str,
        run_dir: str,
    ) -> list[dict[str, Any]]:  # pragma: no cover — network + subprocess path
        root = Path(self.lme_checkout)
        eval_py = root / "third_party/longmemeval/src/evaluation/evaluate_qa.py"
        hyp_path = Path(hypotheses_path)
        ref_path = Path(references_path)
        run_dir_p = Path(run_dir)
        log_path = run_dir_p / "eval.log"
        with open(log_path, "w") as logf:
            subprocess.run(
                [
                    sys.executable, str(eval_py),
                    self.name, str(hyp_path), str(ref_path),
                ],
                stdout=logf, stderr=subprocess.STDOUT,
                cwd=str(root), check=True,
            )
        results_path = hyp_path.with_name(hyp_path.name + f".eval-results-{self.name}")
        return _parse_per_question_results(results_path)


def _parse_per_question_results(path: Path) -> list[dict[str, Any]]:  # pragma: no cover
    """Parse LME's upstream per-question results file.

    Upstream evaluate_qa.py writes one JSON record per line of the form
    `{question_id, autoeval_label, ...}` where autoeval_label is "1" or
    "0" (correct / wrong). We normalize to {question_id, correct, label,
    raw}.
    """
    out: list[dict[str, Any]] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = rec.get("question_id") or rec.get("qid")
        label = str(rec.get("autoeval_label", rec.get("label", "")))
        correct = label in ("1", "true", "True", "correct", "yes")
        out.append({
            "question_id": qid,
            "correct": correct,
            "label": label,
            "raw": rec,
        })
    return out


@dataclass
class FakeJudge:
    """Deterministic test judge with a NAIVE structural rule:

      A hypothesis is "correct" iff at least one of the agent's
      selected_turn_ids belongs to one of the instance's
      answer_session_ids.

    This is not semantic; it's a proxy that lets us unit-test the
    eval-wrapper end-to-end without API keys. Outcome.correct under
    FakeJudge is testing the *plumbing*, not the agent's intelligence.
    """

    name: str = "fake-judge-v1"
    references: dict[str, dict[str, Any]] | None = None
    selected_by_qid: dict[str, list[str]] | None = None

    def judge(
        self,
        *,
        hypotheses_path: str,
        references_path: str,
        run_dir: str,
    ) -> list[dict[str, Any]]:
        # FakeJudge reads from in-memory state set by RealEval before invoking
        # — we don't actually need the files for the rule, but we still write
        # a results file for parity with the real path.
        refs = self.references or json.loads(Path(references_path).read_text())
        if isinstance(refs, list):
            refs = {r["question_id"]: r for r in refs}
        selected = self.selected_by_qid or {}
        out = []
        for qid, ref in refs.items():
            gold = set(ref.get("answer_session_ids", []) or [])
            sel_tids = selected.get(qid, [])
            hit = False
            for tid in sel_tids:
                sid = tid.split("#", 1)[0] if "#" in tid else tid
                if sid in gold:
                    hit = True
                    break
            # abstention questions: "correct" iff selected nothing useful AND
            # we therefore abstained. Mock-faithful approximation.
            if not gold:
                correct = (len(sel_tids) == 0)
            else:
                correct = hit
            out.append({
                "question_id": qid,
                "correct": correct,
                "label": "fake-1" if correct else "fake-0",
                "raw": {"selected_count": len(sel_tids), "gold_count": len(gold)},
            })
        # mirror upstream by writing a results file
        run_dir_p = Path(run_dir)
        run_dir_p.mkdir(parents=True, exist_ok=True)
        (run_dir_p / "fake_per_question_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in out) + "\n"
        )
        return out


# ============================================================================
# RealEval — the wrapper
# ============================================================================


def _canonical_qtype(qid: str, instance: dict | None) -> str:
    """Question_type without the _abs suffix; sourced from instance when
    available, otherwise inferred from the qid (synthetic-fixture form
    uses underscored type prefix)."""
    if instance is not None and "question_type" in instance:
        return str(instance["question_type"])
    # fallback: try prefix match against canonical types
    base = qid[: -len("_abs")] if qid.endswith("_abs") else qid
    for t in (
        "single-session-user", "single-session-assistant",
        "single-session-preference", "multi-session",
        "temporal-reasoning", "knowledge-update",
    ):
        if base.startswith(t.replace("-", "_")):
            return t
    return "unknown"


@dataclass
class RealEval:
    """The eval backend. Composes a Reader + Judge; runs them over a list
    of instances and returns Outcome records ready for diagnose."""

    reader: Reader
    judge: Judge
    signal: str = "embedding"           # match rag-dense
    token_budget: int = 2500            # match rag-dense-turn cell
    min_token_length: int = 4
    min_session_cooccurrence: int = 2
    max_doc_freq_fraction: float = 0.25

    def run_on_split(
        self,
        instances: Iterable[dict[str, Any]],
        *,
        run_dir: str | Path,
    ) -> EvalResult:
        run_dir_p = Path(run_dir)
        run_dir_p.mkdir(parents=True, exist_ok=True)

        instances = list(instances)
        traces: dict[str, Any] = {}
        hypotheses: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        errors: dict[str, str] = {}

        for inst in instances:
            qid = inst["question_id"]
            try:
                trace = agent_retrieve(
                    inst,
                    signal=self.signal,
                    token_budget=self.token_budget,
                    min_token_length=self.min_token_length,
                    min_session_cooccurrence=self.min_session_cooccurrence,
                    max_doc_freq_fraction=self.max_doc_freq_fraction,
                )
                ctx_text = trace.context.text
                hyp = self.reader.answer(
                    context=ctx_text, question=inst["question"], question_id=qid,
                )
            except Exception as e:  # noqa: BLE001 — runtime path; carried to outcome
                trace = None
                hyp = ""
                errors[qid] = f"{type(e).__name__}: {e}"

            traces[qid] = trace
            hypotheses.append({"question_id": qid, "hypothesis": hyp})
            references.append({
                "question_id": qid,
                "question": inst["question"],
                "answer": inst.get("answer", ""),
                "question_type": _canonical_qtype(qid, inst),
                "answer_session_ids": list(inst.get("answer_session_ids", [])),
                "is_abstention": qid.endswith("_abs"),
            })

        hyp_path = run_dir_p / "hypotheses.jsonl"
        hyp_path.write_text(
            "\n".join(json.dumps(h) for h in hypotheses) + "\n"
        )
        ref_path = run_dir_p / "references.json"
        ref_path.write_text(json.dumps(references, indent=2) + "\n")

        # Allow FakeJudge to read from in-memory state without round-tripping
        # through the gold ref file (still writes it for parity / audit).
        if isinstance(self.judge, FakeJudge):
            self.judge.references = {r["question_id"]: r for r in references}
            self.judge.selected_by_qid = {
                qid: list((traces.get(qid).context.meta.get("selected_turn_ids", [])
                          if traces.get(qid) is not None else []))
                for qid in [i["question_id"] for i in instances]
            }

        judgments = self.judge.judge(
            hypotheses_path=str(hyp_path),
            references_path=str(ref_path),
            run_dir=str(run_dir_p),
        )

        # Build Outcome records joining trace + verdict.
        verdict_by_qid = {j["question_id"]: j for j in judgments}
        ref_by_qid = {r["question_id"]: r for r in references}
        outcomes: list[Outcome] = []
        for inst in instances:
            qid = inst["question_id"]
            trace = traces.get(qid)
            verdict = verdict_by_qid.get(qid, {"correct": False, "label": "missing", "raw": None})
            ref = ref_by_qid[qid]
            meta = trace.context.meta if trace is not None else {}

            o = Outcome(
                question_id=qid,
                question_type=ref["question_type"],
                is_abstention=ref["is_abstention"],
                answer_session_ids=tuple(ref["answer_session_ids"]),
                correct=bool(verdict["correct"]),
                judge_label=str(verdict.get("label", "")),
                judge_raw=verdict.get("raw"),
                hypothesis=next(
                    (h["hypothesis"] for h in hypotheses if h["question_id"] == qid),
                    "",
                ),
                signal=str(meta.get("signal", self.signal)),
                selected_turn_ids=tuple(meta.get("selected_turn_ids", ())),
                n_seeds=int(meta.get("n_seeds", 0)),
                n_expanded=int(meta.get("n_expanded", 0)),
                truncated=bool(trace.context.truncated) if trace is not None else False,
                running_tokens=int(meta.get("running_tokens", 0)),
                decisions=tuple(meta.get("decisions", ())),
                scores=dict(meta.get("scores", {})),
                ranked=tuple(meta.get("ranked", ())),
                applied_transforms=tuple(meta.get("applied_transforms", ())),
                run_id=(trace.run_id if trace is not None else ""),
                error=errors.get(qid),
            )
            outcomes.append(o)

        # Aggregate computed from outcomes (single source of truth).
        per_type_correct: dict[str, int] = defaultdict(int)
        per_type_total: dict[str, int] = defaultdict(int)
        n_truncated = 0
        n_errors = 0
        total_tokens = 0
        for o in outcomes:
            per_type_total[o.question_type] += 1
            if o.correct:
                per_type_correct[o.question_type] += 1
            if o.truncated:
                n_truncated += 1
            if o.error:
                n_errors += 1
            total_tokens += o.running_tokens

        aggregate = {
            "version": REGIMES_RUN_VERSION,
            "n": len(outcomes),
            "overall_accuracy": sum(1 for o in outcomes if o.correct) / len(outcomes)
                if outcomes else 0.0,
            "per_type_accuracy": {
                t: per_type_correct[t] / per_type_total[t] for t in sorted(per_type_total)
            },
            "n_truncated": n_truncated,
            "n_errors": n_errors,
            "mean_context_tokens": total_tokens / len(outcomes) if outcomes else 0.0,
        }
        (run_dir_p / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

        return EvalResult(
            outcomes=outcomes,
            aggregate=aggregate,
            backend="real",
            run_dir=str(run_dir_p),
            config={
                "signal": self.signal,
                "token_budget": self.token_budget,
                "reader": self.reader.name,
                "judge": self.judge.name,
                "min_token_length": self.min_token_length,
                "min_session_cooccurrence": self.min_session_cooccurrence,
                "max_doc_freq_fraction": self.max_doc_freq_fraction,
            },
        )
