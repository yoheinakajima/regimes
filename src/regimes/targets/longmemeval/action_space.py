"""LongMemEval action space: three transform-type seams.

The widened action space supports three transform types, each with its
own call shape, static-gate whitelist, and install/revert seam:

  score-transform         — reweight scores before assembly
  assembly-transform      — reorder/filter the selected turn list
  reader-prompt-transform — edit reader prompt fragments

Selective drafting routes diagnosed regimes to the appropriate type:
  budget-truncation / assembly-crowding → score-transform OR assembly-transform
  assemble-internal                     → reader-prompt-transform
  retrieval-signal-gap                  → wall (no type)

All three types flow through the same generic gate chain (static →
sandbox → eval-diff → confirm). Per-type configuration (signature,
whitelist, call_fn, value_validator) is injected via the TransformType
descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from regimes.agent import reader_transforms as _reader_transforms
from regimes.agent import transforms as _agent_transforms
from regimes.eval.types import EvalResult
from regimes.loop import gates as _gates
from regimes.loop.hypothesize import DraftedTransform, StubAuthor
from regimes.target import DraftedChange, EvalDiff, PromotionDecision, SandboxResult, StaticResult

from regimes.targets.longmemeval.transform_types import (
    ALL_TRANSFORM_TYPES,
    ASSEMBLY_TRANSFORM,
    READER_PROMPT_TRANSFORM,
    REGIME_TO_TYPES,
    SCORE_TRANSFORM,
    TransformType,
)


# LongMemEval score-transform signature/whitelist pins (kept for
# backward compatibility with callers that import these constants).
LONGMEMEVAL_SIGNATURE_PARAMS: tuple[str, ...] = SCORE_TRANSFORM.signature_params
LONGMEMEVAL_IMPORT_WHITELIST: frozenset[str] = SCORE_TRANSFORM.import_whitelist

LONGMEMEVAL_PER_TYPE_FLOORS: dict[str, float] = {"multi-session": 0.0}


# ---------------------------------------------------------------------------
# Per-type install/revert registries
# ---------------------------------------------------------------------------

# Assembly-transform has its own in-process pipeline (same pattern as
# regimes.agent.transforms). The reader-prompt-transform now installs
# into the shared `regimes.agent.reader_transforms` pipeline that BOTH
# the mock eval and the real reader path apply — so an installed
# reader-prompt-transform actually changes the prompt the reader sees
# during eval-diff (previously it installed into a local dict no eval
# path ever read, so it could never affect the outcome).

import threading

_ASSEMBLY_LOCK = threading.Lock()
_ASSEMBLY_PIPELINE: dict[str, Callable] = {}


def _install_assembly(name: str, fn: Callable) -> None:
    with _ASSEMBLY_LOCK:
        _ASSEMBLY_PIPELINE[name] = fn


def _revert_assembly(name: str) -> None:
    with _ASSEMBLY_LOCK:
        _ASSEMBLY_PIPELINE.pop(name, None)


def _install_reader(name: str, fn: Callable) -> None:
    _reader_transforms.promote(name, fn)


def _revert_reader(name: str) -> None:
    _reader_transforms.revert(name)


def clear_all_pipelines() -> None:
    """Test isolation helper."""
    _agent_transforms.clear()
    _reader_transforms.clear()
    with _ASSEMBLY_LOCK:
        _ASSEMBLY_PIPELINE.clear()


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------


@dataclass
class LongMemEvalActionSpace:
    """Implements `regimes.target.ActionSpace` for LongMemEval.

    Supports three transform types via selective drafting. The `author`
    produces a DraftedChange whose source matches the type selected for
    the diagnosed regime."""

    author: Any = field(default_factory=StubAuthor)
    per_type_floors: Mapping[str, float] = field(
        default_factory=lambda: dict(LONGMEMEVAL_PER_TYPE_FLOORS)
    )
    overall_floor_delta: float = 0.0
    confirm_threshold: float = 0.0
    n_probe_outcomes: int = 5
    sandbox_time_budget_s: float = 2.0

    # The active transform type for the current draft cycle. Set by
    # draft() and read by the gate methods.
    _active_type: TransformType = field(default=SCORE_TRANSFORM, init=False, repr=False)

    # ---- authoring ---------------------------------------------------------

    def draft(self, *, dominant_regime: str, failures: Sequence[Any]) -> DraftedChange:
        """Selectively draft based on the diagnosed regime."""
        transform_type = self._select_type(dominant_regime)
        self._active_type = transform_type

        if hasattr(self.author, "draft_typed"):
            d = self.author.draft_typed(
                dominant_regime=dominant_regime,
                failures=list(failures),
                transform_type=transform_type.name,
            )
        else:
            d = self.author.draft(
                dominant_regime=dominant_regime,
                failures=list(failures),
            )
        return DraftedChange(
            name=d.name,
            source=d.source,
            target_regime=d.target_regime,
            author=d.author,
            rationale=d.rationale,
            transform_type=transform_type.name,
        )

    def _select_type(self, regime: str) -> TransformType:
        """Route regime → transform type."""
        eligible = REGIME_TO_TYPES.get(regime, ())
        if not eligible:
            return SCORE_TRANSFORM
        return ALL_TRANSFORM_TYPES[eligible[0]]

    # ---- gates -------------------------------------------------------------

    def static_gate(self, source: str) -> StaticResult:
        t = self._active_type
        return _gates.static_gate(
            source,
            expected_fn=t.expected_fn,
            signature_params=t.signature_params,
            import_whitelist=t.import_whitelist,
        )

    def compile(self, source: str) -> Callable:
        return _gates.compile_transform(source, expected_fn=self._active_type.expected_fn)

    def sandbox_gate(
        self, fn: Callable, *, probes: Sequence[Mapping[str, Any]]
    ) -> SandboxResult:
        t = self._active_type

        def _call(f: Callable, probe: Mapping[str, Any]) -> Any:
            return t.call_fn(f, probe)

        def _validate(out: Mapping[str, Any]) -> None:
            # The generic sandbox gate calls value_validator(out); we need
            # the probe context for our validators. We capture the current
            # probe in the closure below via the wrapper in sandbox_gate.
            pass

        # Use a custom wrapper that threads probe context through
        return self._sandbox_with_probe_context(fn, list(probes), t)

    def _sandbox_with_probe_context(
        self, fn: Callable, probes: list[dict[str, Any]], t: TransformType
    ) -> SandboxResult:
        """Run sandbox gate with per-type call_fn and value_validator that
        receives probe context."""
        import time
        reasons: list[str] = []
        t0 = time.perf_counter()
        n_done = 0
        try:
            for p in probes:
                input_keys = self._probe_input_keys(p, t)
                out = t.call_fn(fn, p)

                # Type-specific return shape check (list for assembly, dict for others)
                if t.name == "assembly-transform":
                    if not isinstance(out, list):
                        reasons.append(
                            f"non-list return at probe {n_done}: {type(out).__name__}"
                        )
                        break
                else:
                    if not isinstance(out, dict):
                        reasons.append(
                            f"non-dict return at probe {n_done}: {type(out).__name__}"
                        )
                        break
                    extra = set(out) - input_keys
                    if extra:
                        reasons.append(
                            f"introduced unknown keys at probe {n_done}: "
                            f"{sorted(extra)[:3]}"
                        )
                        break

                try:
                    t.value_validator(out, p)
                except (TypeError, ValueError) as e:
                    reasons.append(f"value-validator failed at probe {n_done}: {e}")
                    break
                n_done += 1
                if time.perf_counter() - t0 > self.sandbox_time_budget_s:
                    reasons.append(f"time budget exceeded after {n_done} probes")
                    break
        except Exception as e:  # noqa: BLE001
            reasons.append(f"raised at probe {n_done}: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - t0
        return SandboxResult(
            passed=len(reasons) == 0 and n_done == len(probes),
            reasons=tuple(reasons),
            n_probed=n_done,
            elapsed_s=elapsed,
        )

    def _probe_input_keys(self, probe: Mapping[str, Any], t: TransformType) -> set:
        """Extract the expected output keys from the probe for the "no
        unknown keys" check."""
        key = t.probe_input_key
        val = probe.get(key, {})
        if isinstance(val, dict):
            return set(val.keys())
        if isinstance(val, (list, tuple)):
            return set(val)
        return set()

    def build_probes(self, baseline: EvalResult) -> list[dict[str, Any]]:
        """Build probes shaped for the active transform type."""
        t = self._active_type
        probes: list[dict[str, Any]] = []
        for o in baseline.outcomes[: self.n_probe_outcomes]:
            if t.name == "score-transform":
                probes.append({
                    "scores": dict(o.scores),
                    "question": "",
                    "question_date": "",
                })
            elif t.name == "assembly-transform":
                probes.append({
                    "selected_turns": list(o.selected_turn_ids),
                    "scores": dict(o.scores),
                    "question": "",
                    "question_date": "",
                })
            elif t.name == "reader-prompt-transform":
                probes.append({
                    "prompt_parts": {
                        "context": " ".join(o.selected_turn_ids) if o.selected_turn_ids else "ctx",
                        "instruction": "Answer the question based on context.",
                    },
                    "question": "",
                    "question_date": "",
                })
        return probes

    # ---- pipeline install / revert ----------------------------------------

    def install(self, name: str, fn: Callable) -> None:
        t = self._active_type
        if t.name == "score-transform":
            _agent_transforms.promote(name, fn)
        elif t.name == "assembly-transform":
            _install_assembly(name, fn)
        elif t.name == "reader-prompt-transform":
            _install_reader(name, fn)

    def revert(self, name: str) -> None:
        # Revert from all pipelines (safe: only one will have the name)
        _agent_transforms.revert(name)
        _revert_assembly(name)
        _reader_transforms.revert(name)

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
            install=self.install, revert=self.revert,
        )

    def promotion_decision(self, diff: EvalDiff) -> PromotionDecision:
        return _gates.promotion_decision(
            diff,
            per_type_floors=self.per_type_floors,
            overall_floor_delta=self.overall_floor_delta,
        )
