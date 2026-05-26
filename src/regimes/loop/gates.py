"""The four lifecycle gates for a drafted transform.

  static_gate(src)              — AST-only checks; no execution.
  sandbox_gate(fn, instances)   — execute on a handful of instances
                                   under a soft time budget; assert
                                   well-formed dict over the input keys.
  eval_diff(fn, eval, instances, baseline)
                                — run the full eval with the transform
                                   installed; compute deltas vs baseline.
  promotion_decision(diff, ...) — deterministic eligibility rule.

Each gate emits its result as an event (see behaviors.py). Gates themselves
are pure functions of their inputs so they're easy to unit-test.

Static gate scope:
  - parse OK
  - top-level def matches the exact signature
  - imports limited to the whitelist (only `math`)
  - no banned attribute access on dangerous targets
  - no banned identifiers (open, __import__, eval, exec, ...)
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from regimes.agent import transforms as _agent_transforms
from regimes.eval.types import EvalResult
from regimes.loop.regimes import classify, histogram


# ===========================================================================
# Static gate
# ===========================================================================


IMPORT_WHITELIST = frozenset({"math"})

BANNED_NAMES = frozenset({
    "open", "input", "eval", "exec", "compile", "exit", "quit",
    "globals", "locals", "vars", "breakpoint",
    "__import__", "getattr", "setattr", "delattr", "hasattr",
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__loader__", "__spec__",
    "__file__", "__name__",
})

BANNED_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__code__", "__closure__",
    "__getattribute__", "__getattr__", "__import__",
    "__loader__", "__spec__", "__file__",
})

REQUIRED_SIGNATURE_PARAMS = ("scores", "graph", "question", "question_date")


@dataclass(frozen=True)
class StaticResult:
    passed: bool
    reasons: tuple[str, ...]
    fn_name: str = "transform"


def static_gate(
    source: str,
    *,
    expected_fn: str = "transform",
    signature_params: tuple[str, ...] | None = None,
    import_whitelist: frozenset[str] | None = None,
) -> StaticResult:
    """AST-based static analysis. Returns reasons rather than raising.

    `signature_params` and `import_whitelist` default to the
    LongMemEval-shaped score-transform constants
    (`REQUIRED_SIGNATURE_PARAMS`, `IMPORT_WHITELIST`) so existing
    callers see identical behavior; concrete ActionSpaces pass their
    own when their action-space differs (e.g. a SQL prompt-edit
    signature in Phase 2)."""
    if signature_params is None:
        signature_params = REQUIRED_SIGNATURE_PARAMS
    if import_whitelist is None:
        import_whitelist = IMPORT_WHITELIST

    reasons: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return StaticResult(passed=False, reasons=(f"syntax-error: {e}",))

    # Top-level must contain exactly one def transform(...) and optional imports.
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    other = [n for n in tree.body
             if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.Import, ast.ImportFrom))]

    if len(defs) != 1 or defs[0].name != expected_fn:
        reasons.append(f"top-level must define a single function `{expected_fn}`")
    if any(isinstance(n, ast.AsyncFunctionDef) for n in defs):
        reasons.append("async transforms not permitted")
    if other:
        # The static-gate's safety story is "AST is exactly: imports + one def".
        # Top-level statements would let arbitrary code run at import time.
        kinds = sorted({type(n).__name__ for n in other})
        reasons.append(f"non-import non-def top-level statements: {kinds}")

    for imp in imports:
        names = [a.name.split(".")[0] for a in imp.names] if isinstance(imp, ast.Import) \
            else [imp.module.split(".")[0]] if imp.module else []
        for n in names:
            if n not in import_whitelist:
                reasons.append(f"import outside whitelist: {n!r}")

    if defs:
        fn = defs[0]
        params = [a.arg for a in fn.args.args]
        if tuple(params) != signature_params:
            reasons.append(
                f"signature mismatch: got {params!r}, "
                f"expected {list(signature_params)!r}"
            )

    # Walk the whole tree for banned names + attribute access.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            reasons.append(f"banned name: {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            reasons.append(f"banned attribute: {node.attr!r}")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_NAMES:
                reasons.append(f"banned call: {f.id!r}()")

    # dedupe while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for r in reasons:
        if r not in seen:
            deduped.append(r)
            seen.add(r)
    return StaticResult(passed=len(deduped) == 0, reasons=tuple(deduped))


def compile_transform(source: str, *, expected_fn: str = "transform") -> Callable:
    """Compile the source string to a callable. Caller MUST run static_gate
    first; compile_transform does NOT re-validate."""
    ns: dict[str, Any] = {}
    code = compile(source, filename=f"<transform:{expected_fn}>", mode="exec")
    exec(code, ns, ns)  # noqa: S102 — gated by static analysis
    fn = ns.get(expected_fn)
    if not callable(fn):
        raise ValueError(f"compiled source has no callable {expected_fn!r}")
    return fn


# ===========================================================================
# Sandbox gate
# ===========================================================================


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    reasons: tuple[str, ...]
    n_probed: int = 0
    elapsed_s: float = 0.0


def sandbox_gate(
    fn: Callable,
    *,
    probes: list[dict[str, Any]],
    time_budget_s: float = 2.0,
) -> SandboxResult:
    """Run the compiled transform on a handful of recorded score dicts.

    `probes` is a list of {"scores": dict, "question": str,
    "question_date": str}. Pulled from the BASELINE outcomes by the
    caller. We assert: no exception, returns a dict, keys are a subset
    of the input, values are floats, total wall time under budget."""
    reasons: list[str] = []
    t0 = time.perf_counter()
    n_done = 0
    try:
        for p in probes:
            input_scores = dict(p.get("scores", {}))
            out = fn(input_scores, None, p.get("question", ""),
                     p.get("question_date", ""))
            if not isinstance(out, dict):
                reasons.append(
                    f"non-dict return at probe {n_done}: {type(out).__name__}"
                )
                break
            extra = set(out) - set(input_scores)
            if extra:
                reasons.append(
                    f"introduced unknown turn_ids at probe {n_done}: "
                    f"{sorted(extra)[:3]}"
                )
                break
            try:
                {tid: float(v) for tid, v in out.items()}
            except (TypeError, ValueError) as e:
                reasons.append(f"non-float values at probe {n_done}: {e}")
                break
            n_done += 1
            if time.perf_counter() - t0 > time_budget_s:
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


# ===========================================================================
# Eval-diff gate
# ===========================================================================


@dataclass(frozen=True)
class EvalDiff:
    overall_before: float
    overall_after: float
    overall_delta: float
    per_type_delta: dict[str, float]
    regime_before: dict[str, int]
    regime_after: dict[str, int]
    target_regime: str
    target_delta: int           # negative = shrank (good)
    # Per-question regime change. Each row is (qid, from, to).
    transitions: tuple[tuple[str, str, str], ...] = ()

    def regression_per_type(self) -> dict[str, float]:
        """Subset of per_type_delta whose value is negative."""
        return {t: d for t, d in self.per_type_delta.items() if d < 0}


def _regime_counts(result: EvalResult) -> dict[str, int]:
    rows = histogram(result.outcomes)
    return {r.regime: r.count for r in rows}


def _per_question_regime(result: EvalResult) -> dict[str, str]:
    """Per-qid regime name. Failures keep their regime; correct answers
    are labeled 'correct' so eval-diff can detect regime->correct
    transitions (the actionable kind)."""
    out: dict[str, str] = {}
    for o in result.outcomes:
        if o.correct:
            out[o.question_id] = "correct"
        else:
            out[o.question_id] = classify(o).name
    return out


def eval_diff(
    *,
    fn: Callable,
    fn_name: str,
    target_regime: str,
    baseline: EvalResult,
    eval_backend,
    instances: list[Any],
) -> EvalDiff:
    """Run `eval_backend.run_on_split(instances)` once with `fn` promoted,
    once with it reverted, and return the resulting EvalDiff. Use of
    promote/revert keeps the agent's seam single-channel."""
    _agent_transforms.promote(fn_name, fn)
    try:
        after = eval_backend.run_on_split(instances)
    finally:
        _agent_transforms.revert(fn_name)

    r_before = _regime_counts(baseline)
    r_after = _regime_counts(after)
    per_type_before = baseline.per_type_accuracy()
    per_type_after = after.per_type_accuracy()
    per_type_delta = {
        t: per_type_after.get(t, 0.0) - per_type_before.get(t, 0.0)
        for t in sorted(set(per_type_before) | set(per_type_after))
    }
    before_qregime = _per_question_regime(baseline)
    after_qregime = _per_question_regime(after)
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


# ===========================================================================
# Promotion decision
# ===========================================================================


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...] = field(default=())


# The LongMemEval-default per-type promotion floor: multi-session must
# not regress. Phase 1 keeps this as the default so existing callers see
# identical behavior; concrete targets (LongMemEvalTarget) pass their own
# map through ActionSpace.promotion_decision.
_DEFAULT_PER_TYPE_FLOORS: dict[str, float] = {"multi-session": 0.0}


def promotion_decision(
    diff: EvalDiff,
    *,
    per_type_floors: Mapping[str, float] | None = None,
    overall_floor_delta: float = 0.0,
) -> PromotionDecision:
    """Deterministic eligibility rule. Promotion-eligible iff:

      (a) the targeted regime SHRANK (target_delta < 0),
      (b) no question_type in `per_type_floors` regressed past its floor,
      (c) overall accuracy did not regress past `overall_floor_delta`.

    `per_type_floors` defaults to {"multi-session": 0.0} — the
    LongMemEval-shaped rule. The fixed-dict default preserves the
    pre-refactor decision exactly for any caller that doesn't pass it.
    """
    floors = _DEFAULT_PER_TYPE_FLOORS if per_type_floors is None else per_type_floors
    reasons: list[str] = []
    if diff.target_delta >= 0:
        reasons.append(
            f"target regime did not shrink: target_delta={diff.target_delta}"
        )
    for qtype, floor in floors.items():
        d = diff.per_type_delta.get(qtype)
        if d is not None and d < floor:
            reasons.append(f"{qtype} regressed by {d:+.4f}")
    if diff.overall_delta < overall_floor_delta:
        reasons.append(f"overall regressed by {diff.overall_delta:+.4f}")
    return PromotionDecision(eligible=len(reasons) == 0, reasons=tuple(reasons))
