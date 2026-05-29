"""SqlActionSpace: the prompt-transform pipeline + the SQL-shaped
gates.

Static-gate is reused verbatim from `regimes.loop.gates.static_gate`
— that function already takes `signature_params` and `import_whitelist`
as kwargs, so the SQL signature pin
`(prompt_parts, question, schema_meta)` and the
`{math, string}` whitelist drop in without modification.

Sandbox-gate is reimplemented here. The generic gates.sandbox_gate
hard-codes the score-transform call shape
`fn(scores, None, question, question_date)` and a float-coercion check
on the returned values — both LME-specific. We mirror the gate's
shape (StaticResult/SandboxResult, n_probed, elapsed_s) so the loop's
behaviors see the same protocol.

Eval-diff is reimplemented here too. The generic gates.eval_diff
delegates to `regimes.loop.regimes.classify` directly to count
`regime_before` / `regime_after`, which is LME-specific (see
docs/PHASE1_5_LEAKS.md). We compute the diff with the SQL taxonomy
so promotion can see the regime actually shrinking.

Promotion-decision delegates to the generic
`gates.promotion_decision` with SQL-appropriate `per_type_floors`
(empty by default — v1 SQL fixture is too small to defend a
per-type floor reliably).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from regimes.eval.types import EvalResult
from regimes.loop import gates as _gates
from regimes.target import (
    DraftedChange,
    EvalDiff,
    PromotionDecision,
    SandboxResult,
    StaticResult,
)
from regimes.targets.sql import prompt_transforms as _pipeline
from regimes.targets.sql.hypothesize import StubSqlAuthor
from regimes.targets.sql.taxonomy import SqlTaxonomy


SQL_SIGNATURE_PARAMS: tuple[str, ...] = ("prompt_parts", "question", "schema_meta")
SQL_IMPORT_WHITELIST: frozenset[str] = frozenset({"math", "string"})


@dataclass
class SqlActionSpace:
    """Implements `regimes.target.ActionSpace` for the SQL target.

    The SQL action space is "append a Python prompt-transform to the
    sql_agent.prompt_pipeline seam" — the analog of LME's score-
    transform pipeline."""

    author: Any = field(default_factory=StubSqlAuthor)
    taxonomy: SqlTaxonomy = field(default_factory=SqlTaxonomy)
    signature_params: tuple[str, ...] = SQL_SIGNATURE_PARAMS
    import_whitelist: frozenset[str] = SQL_IMPORT_WHITELIST
    expected_fn: str = "transform"
    per_type_floors: Mapping[str, float] = field(default_factory=dict)
    overall_floor_delta: float = 0.0
    n_probe_outcomes: int = 5
    sandbox_time_budget_s: float = 2.0

    # ---- authoring --------------------------------------------------------

    def draft(self, *, dominant_regime: str, failures: Sequence[Any]) -> DraftedChange:
        return self.author.draft(
            dominant_regime=dominant_regime, failures=list(failures),
        )

    # ---- gates ------------------------------------------------------------

    def static_gate(self, source: str) -> StaticResult:
        return _gates.static_gate(
            source,
            expected_fn=self.expected_fn,
            signature_params=self.signature_params,
            import_whitelist=self.import_whitelist,
        )

    def compile(self, source: str) -> Callable:
        return _gates.compile_transform(source, expected_fn=self.expected_fn)

    def sandbox_gate(
        self, fn: Callable, *, probes: Sequence[Mapping[str, Any]]
    ) -> SandboxResult:
        """Run the compiled prompt-transform against the recorded
        `prompt_parts` from a handful of baseline outcomes. Asserts:
        no exception, returns a dict, keys are a subset of the input
        prompt_parts dict, total wall time under the soft budget."""
        reasons: list[str] = []
        t0 = time.perf_counter()
        n_done = 0
        try:
            for p in probes:
                input_parts = dict(p.get("prompt_parts", {}))
                out = fn(
                    input_parts,
                    p.get("question", ""),
                    dict(p.get("schema_meta", {})),
                )
                if not isinstance(out, dict):
                    reasons.append(
                        f"non-dict return at probe {n_done}: {type(out).__name__}"
                    )
                    break
                extra = set(out) - set(input_parts)
                if extra:
                    reasons.append(
                        f"introduced unknown prompt_parts keys at probe "
                        f"{n_done}: {sorted(extra)[:3]}"
                    )
                    break
                n_done += 1
                if time.perf_counter() - t0 > self.sandbox_time_budget_s:
                    reasons.append(f"time budget exceeded after {n_done} probes")
                    break
        except Exception as e:  # noqa: BLE001
            reasons.append(f"raised at probe {n_done}: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - t0
        return SandboxResult(
            passed=len(reasons) == 0 and n_done == len(list(probes)),
            reasons=tuple(reasons),
            n_probed=n_done,
            elapsed_s=elapsed,
        )

    def build_probes(self, baseline: EvalResult) -> list[dict[str, Any]]:
        probes: list[dict[str, Any]] = []
        for o in baseline.outcomes[: self.n_probe_outcomes]:
            probes.append({
                "prompt_parts": dict(getattr(o, "prompt_parts", {})),
                "question": getattr(o, "nl_question", ""),
                "schema_meta": dict(getattr(o, "schema_meta", {})),
            })
        return probes

    # ---- install / revert -------------------------------------------------

    def install(self, name: str, fn: Callable) -> None:
        _pipeline.promote(name, fn)

    def revert(self, name: str) -> None:
        _pipeline.revert(name)

    # ---- eval-diff (SQL-taxonomy aware) -----------------------------------

    def eval_diff(
        self,
        *,
        fn: Callable,
        fn_name: str,
        target_regime: str,
        baseline: EvalResult,
        eval_backend: Any,
        instances: Sequence[Any],
    ) -> EvalDiff:
        """Same structure as `gates.eval_diff` but uses the SQL
        taxonomy's `classify` / `histogram` so promotion can see the
        SQL regime actually shrinking. The generic gates.eval_diff
        delegates to LME's classify, which would return
        'unclassified' on every SqlOutcome — see Phase 1.5 leak notes."""
        self.install(fn_name, fn)
        try:
            after = eval_backend.run_on_split(list(instances))
        finally:
            self.revert(fn_name)

        r_before = self._regime_counts(baseline)
        r_after = self._regime_counts(after)
        per_type_before = baseline.per_type_accuracy()
        per_type_after = after.per_type_accuracy()
        per_type_delta = {
            t: per_type_after.get(t, 0.0) - per_type_before.get(t, 0.0)
            for t in sorted(set(per_type_before) | set(per_type_after))
        }
        before_qregime = self._per_qid_regime(baseline)
        after_qregime = self._per_qid_regime(after)
        transitions = tuple(
            (qid, before_qregime[qid], after_qregime.get(qid, "?"))
            for qid in sorted(before_qregime)
            if after_qregime.get(qid) != before_qregime[qid]
        )

        return EvalDiff(
            overall_before=baseline.overall_accuracy(),
            overall_after=after.overall_accuracy(),
            overall_delta=after.overall_accuracy() - baseline.overall_accuracy(),
            per_type_delta=per_type_delta,
            regime_before=r_before,
            regime_after=r_after,
            target_regime=target_regime,
            target_delta=r_after.get(target_regime, 0) - r_before.get(target_regime, 0),
            transitions=transitions,
        )

    def _regime_counts(self, result: EvalResult) -> dict[str, int]:
        rows = self.taxonomy.histogram(result.outcomes)
        return {r.regime: r.count for r in rows}

    def _per_qid_regime(self, result: EvalResult) -> dict[str, str]:
        out: dict[str, str] = {}
        for o in result.outcomes:
            if o.correct:
                out[o.question_id] = "correct"
            else:
                out[o.question_id] = self.taxonomy.classify(o).name
        return out

    # ---- promotion decision ----------------------------------------------

    def promotion_decision(self, diff: EvalDiff) -> PromotionDecision:
        return _gates.promotion_decision(
            diff,
            per_type_floors=self.per_type_floors,
            overall_floor_delta=self.overall_floor_delta,
        )
