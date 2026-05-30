"""Target interface — the plugin surface the loop drives.

The loop's control flow (`runner.py`, `attribute.py`, the now-generic
parts of `behaviors.py`/`gates.py`) is target-agnostic. Everything that
USED to be LongMemEval-specific in those modules — the score-transform
probe shape, the multi-session regression floor, the regime detectors,
the per-outcome summary, the named-wall recommendation strings — lives
behind this interface and is supplied by a concrete `Target`.

A `Target` bundles four things:

  EvalBackend     run_on_split(instances) -> EvalResult
                  (Today's RealEval / MockEval already implement this.)

  ActionSpace     The "what can we change, and how" surface.
                    draft(...) -> DraftedChange
                    static_gate(source)
                    compile(source)
                    sandbox_gate(fn, probes)
                    build_probes(baseline_result)
                    install(name, fn) / revert(name)
                    promotion_decision(diff)
                  For LongMemEval this is the score-transform pipeline.
                  For SQL (Phase 2) it'll be a prompt-edit pipeline.

  RegimeTaxonomy  The deterministic failure-mode taxonomy.
                    REGIMES()
                    classify(outcome) -> Regime
                    histogram(outcomes) -> list[HistogramRow]
                    is_seam_reachable(name) -> bool
                    name_wall(counts) -> str

  outcome_summary A per-target persistence/audit projection of one
                  Outcome into a JSON-shaped dict. The loop emits this
                  on baseline.recorded so reviewers can verify regime
                  labels against the same signals the detector saw.

`EvalResult` is generic (already). `Outcome` is currently retrieval-
shaped; per the investigation map (docs/PLATFORM_INVESTIGATION.md
§Outcome) the loop's touch points on it are narrow enough that
subclass-by-convention is fine: a target's outcomes need at minimum
`question_id: str`, `question_type: str`, `correct: bool`, and any
extra fields its own taxonomy/summary read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from regimes.eval.types import EvalResult


# ---------------------------------------------------------------------------
# DraftedChange — generalization of DraftedTransform
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftedChange:
    """One authored modification to the target. The interpretation of
    `source` is action-space-specific: for LongMemEval it's a Python
    score-transform function body; for SQL it'd be a prompt edit. The
    loop carries this through the gate chain as opaque text and only
    compiles/runs it via the ActionSpace methods."""

    name: str
    source: str
    target_regime: str
    author: str             # "stub" | "claude-..." | etc.
    rationale: str = ""
    # Which action-space seam this change uses. For LongMemEval this is
    # one of score-transform / assembly-transform / reader-prompt-transform;
    # carried through the gate chain so each transform_log entry can record
    # the seam a candidate exercised (the audit otherwise can't tell a
    # reader-prompt-transform from a score-transform). Optional + defaulted
    # so action spaces that don't distinguish seams (e.g. SQL) are unaffected.
    transform_type: str = ""


# ---------------------------------------------------------------------------
# Gate result shapes — re-exported from gates.py so target.py is the
# single import the LoopContext / behaviors need.
# ---------------------------------------------------------------------------

from regimes.loop.gates import (  # noqa: E402  — re-exports kept here on purpose
    EvalDiff,
    PromotionDecision,
    SandboxResult,
    StaticResult,
)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EvalBackend(Protocol):
    """The minimal eval-backend contract. Returns an EvalResult whose
    outcomes the loop's diagnose step feeds into the taxonomy."""

    def run_on_split(self, instances: Sequence[Any]) -> EvalResult: ...


@runtime_checkable
class RegimeTaxonomy(Protocol):
    """Target-specific failure-mode taxonomy.

    Implementations may simply delegate to module-level functions (the
    LongMemEval taxonomy does — see `targets/longmemeval/taxonomy.py`)."""

    def REGIMES(self) -> dict[str, Any]: ...
    def classify(self, outcome: Any) -> Any: ...
    def histogram(self, outcomes: Sequence[Any]) -> list[Any]: ...
    def is_seam_reachable(self, regime_name: str) -> bool: ...
    def name_wall(self, counts: Mapping[str, int]) -> str: ...
    def format_histogram(
        self, rows: Sequence[Any], *, n_failures: int, n_total: int
    ) -> str: ...


@runtime_checkable
class ActionSpace(Protocol):
    """The set of allowed modifications + the gates that vet them.

    For LongMemEval this is the score-transform pipeline. The four
    gates (static / sandbox / eval-diff / promotion) are reused
    verbatim from `regimes.loop.gates`; per-target configuration
    (signature, import whitelist, probe shape, per-type promotion
    floors) lives on the ActionSpace instance.
    """

    def draft(
        self, *, dominant_regime: str, failures: Sequence[Any]
    ) -> DraftedChange: ...

    def static_gate(self, source: str) -> StaticResult: ...
    def compile(self, source: str) -> Callable: ...
    def sandbox_gate(
        self, fn: Callable, *, probes: Sequence[Mapping[str, Any]]
    ) -> SandboxResult: ...
    def build_probes(self, baseline: EvalResult) -> list[dict[str, Any]]: ...

    def install(self, name: str, fn: Callable) -> None: ...
    def revert(self, name: str) -> None: ...

    def eval_diff(
        self,
        *,
        fn: Callable,
        fn_name: str,
        target_regime: str,
        baseline: EvalResult,
        eval_backend: EvalBackend,
        instances: Sequence[Any],
    ) -> EvalDiff: ...

    def promotion_decision(self, diff: EvalDiff) -> PromotionDecision: ...


@runtime_checkable
class Target(Protocol):
    """The full plugin surface. The loop calls these and nothing else."""

    name: str
    eval_backend: EvalBackend
    action_space: ActionSpace
    taxonomy: RegimeTaxonomy

    def outcome_summary(self, outcome: Any) -> dict[str, Any]: ...


__all__ = [
    "ActionSpace",
    "DraftedChange",
    "EvalBackend",
    "EvalDiff",
    "PromotionDecision",
    "RegimeTaxonomy",
    "SandboxResult",
    "StaticResult",
    "Target",
]
