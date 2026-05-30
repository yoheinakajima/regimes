"""The four lifecycle gates for a drafted transform.

  static_gate(src, ...)         — AST-only checks; no execution.
  sandbox_gate(fn, ...)         — execute on a handful of probes under
                                   a soft time budget; assert
                                   well-formed dict over the input keys.
  eval_diff(fn, eval, ...)      — run the full eval with the transform
                                   installed; compute deltas vs baseline.
  promotion_decision(diff, ...) — deterministic eligibility rule.

Each gate emits its result as an event (see behaviors.py). Gates
themselves are pure functions of their inputs so they're easy to
unit-test.

All four gates are TARGET-AGNOSTIC: target-specific knobs (signature
pin, import whitelist, sandbox call shape, value validator,
install/revert seam, regime taxonomy) are passed in as parameters with
LongMemEval-shaped defaults. The defaults preserve historical
direct-call behavior for tests that exercise the gates standalone.

Static gate scope:
  - parse OK
  - top-level def matches the exact signature
  - imports limited to the whitelist (default: only `math`)
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
from regimes.loop.regimes import classify as _lme_classify
from regimes.loop.regimes import histogram as _lme_histogram


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


# ---- LongMemEval defaults for the sandbox seam. -----------------------
# The sandbox gate has two target-specific surfaces — how the candidate
# is *called* (positional shape from the probe dict) and how its return
# is *validated* (LME score-transforms must return float values; SQL
# prompt-transforms return strings/lists). Both are pluggable; the
# defaults below preserve historical behavior for direct callers.


def _lme_call_fn(fn: Callable, probe: Mapping[str, Any]) -> dict:
    """LongMemEval score-transform call shape:
    `fn(scores, None, question, question_date)`. Matches the seam in
    `regimes.agent.behaviors.behavior_transform_pipeline`."""
    input_scores = dict(probe.get("scores", {}))
    return fn(
        input_scores, None,
        probe.get("question", ""), probe.get("question_date", ""),
    )


def _lme_value_validator(out: Mapping[str, Any]) -> None:
    """LongMemEval score-transform value contract: every value must
    coerce to float. Raises TypeError/ValueError on the first bad cell."""
    {tid: float(v) for tid, v in out.items()}


def _probe_input_keys(probe: Mapping[str, Any]) -> set:
    """The keys the candidate's return MUST be a subset of. For LME
    that's the input scores dict; for SQL it's the input prompt_parts
    dict. Inferred from the probe's first dict-typed value so the
    contract works for both shapes without a target tag."""
    # Preference order matches the established probe shapes:
    #   LME score-transforms:    probe["scores"]      (dict[str, float])
    #   SQL prompt-transforms:   probe["prompt_parts"] (dict[str, Any])
    if "scores" in probe and isinstance(probe["scores"], dict):
        return set(probe["scores"])
    if "prompt_parts" in probe and isinstance(probe["prompt_parts"], dict):
        return set(probe["prompt_parts"])
    # Fallback: pick the first dict value (defensive).
    for v in probe.values():
        if isinstance(v, dict):
            return set(v)
    return set()


def sandbox_gate(
    fn: Callable,
    *,
    probes: list[dict[str, Any]],
    time_budget_s: float = 2.0,
    call_fn: Callable[[Callable, Mapping[str, Any]], dict] | None = None,
    value_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> SandboxResult:
    """Run the compiled transform on a handful of probes.

    `probes` is target-shaped: for LME it's
    {"scores": dict, "question": str, "question_date": str}; for SQL
    it's {"prompt_parts": dict, "question": str, "schema_meta": dict}.
    The shape is decoded by `call_fn` and the input-key set is read off
    the probe via `_probe_input_keys` so the "no new keys" invariant
    works on both.

    Per-target hooks (default = LongMemEval score-transform shape so
    historical callers stay identical):
      call_fn         — how to invoke the candidate from one probe.
      value_validator — additional invariant on the returned dict's
                        values (e.g. float coercion for scores). Raises
                        on failure.

    Invariants always enforced (regardless of target):
      - no exception during call_fn
      - return is a dict
      - return's keys ⊆ input keys (no inventing turn_ids / prompt
        parts the seam doesn't know about)
      - total wall time under `time_budget_s`
    """
    if call_fn is None:
        call_fn = _lme_call_fn
    if value_validator is None:
        value_validator = _lme_value_validator
    reasons: list[str] = []
    t0 = time.perf_counter()
    n_done = 0
    try:
        for p in probes:
            input_keys = _probe_input_keys(p)
            out = call_fn(fn, p)
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
                value_validator(out)
            except (TypeError, ValueError) as e:
                reasons.append(f"value-validator failed at probe {n_done}: {e}")
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
    # `from` and `to` are regime-NAME strings produced by whichever
    # taxonomy the diff was computed against — they are TAXONOMY-LOCAL.
    # The `taxonomy_name` field carries the taxonomy-of-origin so two
    # diffs from different targets can be compared without name
    # collisions (e.g. both taxonomies might one day register an
    # "unclassified" regime).
    transitions: tuple[tuple[str, str, str], ...] = ()
    # Empty string = "no taxonomy tag" (older callers / direct
    # constructions in tests). The action-space path populates it.
    taxonomy_name: str = ""

    def regression_per_type(self) -> dict[str, float]:
        """Subset of per_type_delta whose value is negative."""
        return {t: d for t, d in self.per_type_delta.items() if d < 0}


# Minimal duck-typed protocol the eval_diff machinery uses on a
# RegimeTaxonomy. We don't import `regimes.target.RegimeTaxonomy` here
# to avoid a circular dependency (`target` re-exports from this
# module); the real protocol lives there and is structurally identical.
class _TaxonomyLike:
    def classify(self, outcome: Any) -> Any: ...
    def histogram(self, outcomes: Any) -> Any: ...
    name: str


def _regime_counts(
    result: EvalResult,
    *,
    taxonomy: _TaxonomyLike | None,
) -> dict[str, int]:
    """Regime-name → failure count via `taxonomy.histogram`. Defaults
    to the LongMemEval taxonomy so direct callers see no change."""
    if taxonomy is None:
        rows = _lme_histogram(result.outcomes)
    else:
        rows = taxonomy.histogram(result.outcomes)
    return {r.regime: r.count for r in rows}


def _per_question_regime(
    result: EvalResult,
    *,
    taxonomy: _TaxonomyLike | None,
) -> dict[str, str]:
    """Per-qid regime name. Failures keep their regime; correct answers
    are labeled 'correct' so eval-diff can detect regime→correct
    transitions (the actionable kind). `taxonomy=None` defaults to the
    LongMemEval taxonomy."""
    classify_fn = taxonomy.classify if taxonomy is not None else _lme_classify
    out: dict[str, str] = {}
    for o in result.outcomes:
        if o.correct:
            out[o.question_id] = "correct"
        else:
            out[o.question_id] = classify_fn(o).name
    return out


def eval_diff(
    *,
    fn: Callable,
    fn_name: str,
    target_regime: str,
    baseline: EvalResult,
    eval_backend,
    instances: list[Any],
    install: Callable[[str, Callable], None] | None = None,
    revert: Callable[[str], None] | None = None,
    taxonomy: _TaxonomyLike | None = None,
) -> EvalDiff:
    """Run `eval_backend.run_on_split(instances)` once with `fn` installed,
    once with it reverted, and return the resulting EvalDiff.

    Target-agnostic surfaces (default = LongMemEval shape so historical
    direct callers see no change):
      install         — how to attach the candidate to the target's
                        seam. Default: `regimes.agent.transforms.promote`
                        (the LME score-transform pipeline).
      revert          — the inverse. Default: `regimes.agent.transforms.revert`.
      taxonomy        — used to compute `regime_before` / `regime_after`
                        and the per-qid transitions. Default: the
                        LongMemEval taxonomy module functions.

    With these threaded through, an ActionSpace can delegate a single
    `gates.eval_diff(taxonomy=self.taxonomy, install=self.install,
    revert=self.revert, ...)` call and get target-correct regime
    counts without re-implementing the body."""
    if install is None:
        install = _agent_transforms.promote
    if revert is None:
        revert = _agent_transforms.revert

    install(fn_name, fn)
    try:
        after = eval_backend.run_on_split(instances)
    finally:
        revert(fn_name)

    r_before = _regime_counts(baseline, taxonomy=taxonomy)
    r_after = _regime_counts(after, taxonomy=taxonomy)
    per_type_before = baseline.per_type_accuracy()
    per_type_after = after.per_type_accuracy()
    per_type_delta = {
        t: per_type_after.get(t, 0.0) - per_type_before.get(t, 0.0)
        for t in sorted(set(per_type_before) | set(per_type_after))
    }
    before_qregime = _per_question_regime(baseline, taxonomy=taxonomy)
    after_qregime = _per_question_regime(after, taxonomy=taxonomy)
    transitions = tuple(
        (qid, before_qregime[qid], after_qregime.get(qid, "?"))
        for qid in sorted(before_qregime)
        if after_qregime.get(qid) != before_qregime[qid]
    )

    tax_name = getattr(taxonomy, "name", "") if taxonomy is not None else "longmemeval"

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
        taxonomy_name=tax_name,
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
