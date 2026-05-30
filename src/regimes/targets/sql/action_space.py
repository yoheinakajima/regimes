"""SqlActionSpace: the prompt-transform pipeline + the SQL-shaped
gates.

All four gates now go through `regimes.loop.gates`, which Phase 1.5
made target-agnostic (taxonomy / install / revert / call_fn /
value_validator are kwargs with LongMemEval defaults). The SQL
ActionSpace just passes its own:

  - static_gate:          (sig_params, import_whitelist) → SQL set
  - sandbox_gate:         (call_fn, value_validator) → prompt-transform shape
  - eval_diff:            (install, revert, taxonomy) → SQL pipeline + SqlTaxonomy
  - promotion_decision:   (per_type_floors, overall_floor_delta)

Per-target configuration lives on this ActionSpace instance; the gate
bodies are shared.
"""

from __future__ import annotations

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


def _sql_call_fn(fn: Callable, probe: Mapping[str, Any]) -> dict:
    """SQL prompt-transform call shape:
    `fn(prompt_parts, question, schema_meta)`. Matches the seam in
    `sql_agent.behavior_prompt_pipeline`."""
    return fn(
        dict(probe.get("prompt_parts", {})),
        probe.get("question", ""),
        dict(probe.get("schema_meta", {})),
    )


def _sql_value_validator(out: Mapping[str, Any]) -> None:
    """SQL prompt-transforms can return any JSON-shaped values
    (strings, lists, dicts). No additional invariant beyond "is a
    dict and keys ⊆ input keys" — both enforced by sandbox_gate
    itself."""
    return None


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
    confirm_threshold: float = 0.0
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
        """Delegates to the shared `gates.sandbox_gate` with the SQL
        call_fn (prompt-transform signature) and a no-op value
        validator (prompt_parts hold strings/lists, not floats)."""
        return _gates.sandbox_gate(
            fn,
            probes=list(probes),
            time_budget_s=self.sandbox_time_budget_s,
            call_fn=_sql_call_fn,
            value_validator=_sql_value_validator,
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

    # ---- eval-diff --------------------------------------------------------

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
        """Delegates to the shared `gates.eval_diff` with the SQL
        prompt-transform install/revert seam and the SQL taxonomy.
        Phase 1.5 unblocked this — before the gate was hardcoded to
        LME's classify+pipeline and the SQL target had to re-implement
        the body."""
        return _gates.eval_diff(
            fn=fn, fn_name=fn_name, target_regime=target_regime,
            baseline=baseline, eval_backend=eval_backend,
            instances=list(instances),
            install=self.install, revert=self.revert,
            taxonomy=self.taxonomy,
        )

    # ---- promotion decision ----------------------------------------------

    def promotion_decision(self, diff: EvalDiff) -> PromotionDecision:
        return _gates.promotion_decision(
            diff,
            per_type_floors=self.per_type_floors,
            overall_floor_delta=self.overall_floor_delta,
        )
