"""LongMemEvalTarget — the concrete `Target` for the existing loop.

Bundles the four pieces the loop drives: an `EvalBackend` (whatever
`RealEval` / `MockEval` the caller supplies), a `LongMemEvalActionSpace`
(score-transform pipeline + LME-shaped gates), a `LongMemEvalTaxonomy`
(adapter over the existing module-level functions in
`regimes.loop.regimes`), and an `outcome_summary` projection.

The runner constructs one of these from a backend + author so existing
callers (`run_loop(eval_backend=..., author=...)`) keep working
unchanged."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regimes.eval.types import Outcome
from regimes.loop.hypothesize import StubAuthor
from regimes.target import ActionSpace, EvalBackend, RegimeTaxonomy

from regimes.targets.longmemeval.action_space import LongMemEvalActionSpace
from regimes.targets.longmemeval.outcome_summary import outcome_summary
from regimes.targets.longmemeval.taxonomy import LongMemEvalTaxonomy


@dataclass
class LongMemEvalTarget:
    """Concrete LongMemEval `Target`. Composition only — no logic of its
    own beyond delegating to the four components."""

    eval_backend: EvalBackend
    action_space: ActionSpace = field(default_factory=LongMemEvalActionSpace)
    taxonomy: RegimeTaxonomy = field(default_factory=LongMemEvalTaxonomy)
    name: str = "longmemeval"

    def outcome_summary(self, outcome: Outcome) -> dict[str, Any]:
        return outcome_summary(outcome)


def build_target(
    *,
    eval_backend: EvalBackend,
    author: Any = None,
) -> LongMemEvalTarget:
    """Build a LongMemEvalTarget from an eval backend + an optional
    author. Default author is StubAuthor (deterministic, no keys); the
    runner passes an LLMAuthor here for real-mode."""
    action_space = LongMemEvalActionSpace(
        author=author if author is not None else StubAuthor(),
    )
    return LongMemEvalTarget(
        eval_backend=eval_backend,
        action_space=action_space,
        taxonomy=LongMemEvalTaxonomy(),
    )
