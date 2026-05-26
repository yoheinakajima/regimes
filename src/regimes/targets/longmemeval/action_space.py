"""LongMemEval action space: the score-transform pipeline.

The set of allowed modifications is "append a Python score-transform to
the agent's `regimes.agent.transforms` pipeline". The four gates wrap
the existing `regimes.loop.gates` functions with LongMemEval-specific
configuration (signature, import whitelist, probe shape, per-type
promotion floors).

For Phase 2, a `SQLActionSpace` will implement the same interface with
a different signature pin, different probes (prompt + schema instead of
scores), and a different install/revert target (a SQL agent's prompt
pipeline rather than the score-transform pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from regimes.agent import transforms as _agent_transforms
from regimes.eval.types import EvalResult
from regimes.loop import gates as _gates
from regimes.loop.hypothesize import DraftedTransform, StubAuthor
from regimes.target import DraftedChange, EvalDiff, PromotionDecision, SandboxResult, StaticResult


# LongMemEval score-transform signature/whitelist pins. These used to be
# module-level constants in `regimes.loop.gates`; they're now the
# ActionSpace's per-target configuration. The gates module keeps the
# same default values so external callers see no change.
LONGMEMEVAL_SIGNATURE_PARAMS: tuple[str, ...] = (
    "scores", "graph", "question", "question_date",
)
LONGMEMEVAL_IMPORT_WHITELIST: frozenset[str] = frozenset({"math"})

# Default per-type regression floor for promotion: multi-session must
# not regress. Lifted from `gates.promotion_decision`'s previous
# hard-coded "multi-session" check.
LONGMEMEVAL_PER_TYPE_FLOORS: dict[str, float] = {"multi-session": 0.0}


@dataclass
class LongMemEvalActionSpace:
    """Implements `regimes.target.ActionSpace` for LongMemEval.

    The `author` is what produces a DraftedChange — StubAuthor by
    default for tests/mock-mode; an LLMAuthor is wired in for real-mode
    via the loop runner."""

    author: Any = field(default_factory=StubAuthor)
    signature_params: tuple[str, ...] = LONGMEMEVAL_SIGNATURE_PARAMS
    import_whitelist: frozenset[str] = LONGMEMEVAL_IMPORT_WHITELIST
    per_type_floors: Mapping[str, float] = field(
        default_factory=lambda: dict(LONGMEMEVAL_PER_TYPE_FLOORS)
    )
    overall_floor_delta: float = 0.0
    expected_fn: str = "transform"
    n_probe_outcomes: int = 5
    sandbox_time_budget_s: float = 2.0

    # ---- authoring ---------------------------------------------------------

    def draft(self, *, dominant_regime: str, failures: Sequence[Any]) -> DraftedChange:
        d: DraftedTransform = self.author.draft(
            dominant_regime=dominant_regime, failures=list(failures),
        )
        return DraftedChange(
            name=d.name,
            source=d.source,
            target_regime=d.target_regime,
            author=d.author,
            rationale=d.rationale,
        )

    # ---- gates -------------------------------------------------------------

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
        return _gates.sandbox_gate(
            fn, probes=list(probes), time_budget_s=self.sandbox_time_budget_s,
        )

    def build_probes(self, baseline: EvalResult) -> list[dict[str, Any]]:
        """Pull probe inputs from up to N baseline outcomes — enough to
        exercise the score-transform without re-running a full eval.
        Mirrors the previous inline construction in
        `behavior_sandbox_gate`."""
        probes: list[dict[str, Any]] = []
        for o in baseline.outcomes[: self.n_probe_outcomes]:
            probes.append({
                "scores": dict(o.scores),
                "question": "",
                "question_date": "",
            })
        return probes

    # ---- pipeline install / revert ----------------------------------------

    def install(self, name: str, fn: Callable) -> None:
        _agent_transforms.promote(name, fn)

    def revert(self, name: str) -> None:
        _agent_transforms.revert(name)

    # ---- eval-diff + promotion --------------------------------------------

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
        return _gates.eval_diff(
            fn=fn, fn_name=fn_name, target_regime=target_regime,
            baseline=baseline, eval_backend=eval_backend, instances=list(instances),
        )

    def promotion_decision(self, diff: EvalDiff) -> PromotionDecision:
        return _gates.promotion_decision(
            diff,
            per_type_floors=self.per_type_floors,
            overall_floor_delta=self.overall_floor_delta,
        )
