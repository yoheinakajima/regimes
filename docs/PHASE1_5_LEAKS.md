# Phase 1.5 Cleanup: LME-shape leaks discovered when building the SQL target

The Phase 1 refactor (commit `b94edbe`) moved the LongMemEval couplings
behind the `Target` interface and preserved byte-equal LongMemEval
behavior. Phase 2 (the SQL target, commit `f43516a`) exercised that
interface with a genuinely different action space and taxonomy and
surfaced four places where LongMemEval shape still leaked through
the loop-control code.

**Phase 1.5 (commit on `claude/phase-1-5-leaks`) resolves all four.**
After Phase 1.5 the SQL `ActionSpace` no longer re-implements
`eval_diff` or `sandbox_gate` — both go through the now-target-agnostic
shared gate code via short delegation calls.

---

## Leak 1 — `loop/attribute.py` calls LongMemEval's `classify` directly  ✅ RESOLVED

**Was:** `from regimes.loop.regimes import classify` at the top; all
attribution events for SqlOutcome reported transitions like
`("sql_co_q01", "unclassified", "correct")`.

**Now:** `attribute()` takes an optional `taxonomy=` keyword (default =
LongMemEval, preserves direct-caller behavior). The loop's
`behavior_attribute` passes `lctx.target.taxonomy`. SQL attribution
events now report `("sql_co_q01", "schema-misunderstanding",
"correct")`.

**Fix locations:**
- `src/regimes/loop/attribute.py:24-104` — `Attribution.taxonomy_name`
  field added; `attribute(*, taxonomy=None)` signature; `_per_qid_regime`
  threads it through.
- `src/regimes/loop/behaviors.py:512-516` — `behavior_attribute` now
  calls `_attribute(lctx.baseline, after, taxonomy=lctx.target.taxonomy)`.

**Evidence:** SQL mock loop attribution payloads went from
`[..., "unclassified", "correct"]` to `[..., "schema-misunderstanding",
"correct"]` (etc) — verified by `diff /tmp/sql_before_15/report.json
/tmp/sql_after_15/report.json`. All other fields (events, baseline,
promotions, discards, stopped) byte-equal.

---

## Leak 2 — `gates.eval_diff` / `_per_question_regime` / `_regime_counts` call LME classify/histogram  ✅ RESOLVED (high severity)

**Was:** `gates.eval_diff` hardcoded `regimes.loop.regimes.classify` and
`histogram`, so `regime_before` / `regime_after` were always
LME-flavored regardless of what target produced the outcomes. SQL had
to re-implement the whole `eval_diff` body in
`SqlActionSpace.eval_diff` to get correct counts.

**Now:** `gates.eval_diff(*, taxonomy=None, install=None, revert=None,
...)`. The three new kwargs default to the LongMemEval defaults
(`_lme_classify` / `_lme_histogram` / `regimes.agent.transforms.promote`
/ `regimes.agent.transforms.revert`) so historical direct callers and
the LME action space see no change. The SQL action space passes
`self.taxonomy`, `self.install`, `self.revert`.

**Fix locations:**
- `src/regimes/loop/gates.py:269-292` — `_regime_counts` /
  `_per_question_regime` take a `taxonomy` kwarg.
- `src/regimes/loop/gates.py:294-364` — `eval_diff` takes
  `install` / `revert` / `taxonomy` kwargs with LME defaults.
- `src/regimes/targets/sql/action_space.py:128-149` —
  `SqlActionSpace.eval_diff` shrank from ~50 lines of re-implementation
  to a single 8-line delegation to `_gates.eval_diff`.
- `src/regimes/targets/longmemeval/action_space.py:120-140` —
  `LongMemEvalActionSpace.eval_diff` now passes its `install`/`revert`
  explicitly (no behavior change; just makes the dependency visible).

**Evidence:** The SQL loop still promotes the same 4 stub transforms
(`stub_schema_clarification_hint`, `stub_groupby_hint`,
`stub_fk_join_hint`, `stub_where_hint`) in the same order across 4
iterations, ending in `loop.stopped` with
`reason="no_optimizable_regime_remaining"` — identical to Phase 2.

---

## Leak 3 — `gates.sandbox_gate` hard-codes the LME call shape + float coercion  ✅ RESOLVED (medium severity)

**Was:** `sandbox_gate` called `fn(scores, None, question, question_date)`
verbatim and did a `float()` coercion on the returned values. Both
LME-score-transform-specific. SQL had to re-implement the gate body
in `SqlActionSpace.sandbox_gate`.

**Now:** `gates.sandbox_gate` takes `call_fn` and `value_validator`
kwargs:

```python
def sandbox_gate(fn, *, probes, time_budget_s=2.0,
                 call_fn=None, value_validator=None) -> SandboxResult:
    if call_fn is None: call_fn = _lme_call_fn
    if value_validator is None: value_validator = _lme_value_validator
    ...
```

- `_lme_call_fn` → `fn(scores, None, question, question_date)`
- `_lme_value_validator` → float coercion check
- `_sql_call_fn` (in SqlActionSpace) → `fn(prompt_parts, question, schema_meta)`
- `_sql_value_validator` → no-op (prompt_parts hold strings/lists)

The "no new keys" invariant is now decoded via `_probe_input_keys(probe)`
which checks `probe["scores"]` then `probe["prompt_parts"]` — both
established shapes — so the invariant works on either without a
target tag.

**Fix locations:**
- `src/regimes/loop/gates.py:174-269` — `sandbox_gate` parameterized;
  `_lme_call_fn`, `_lme_value_validator`, `_probe_input_keys` added.
- `src/regimes/targets/sql/action_space.py:54-78,107-116` —
  `_sql_call_fn`, `_sql_value_validator`, and `SqlActionSpace.sandbox_gate`
  shrunk from ~40 lines of re-implementation to an 8-line delegation.

**Note: small test-string change.** The sandbox gate's
"unknown-keys" rejection message used to be target-shaped
(`"introduced unknown turn_ids"` for LME / `"introduced unknown
prompt_parts"` for the SQL re-implementation). It's now the generic
`"introduced unknown keys"`. The gate's structural behavior
(`passed=False`, `n_probed=N`) is identical; only the human-readable
reason text changed. Two test assertions
(`tests/test_loop_gates.py:124`, `tests/test_sql_target.py:277`) were
updated to assert on the generic string. **No loop behavior depends
on the reason text.**

---

## Leak 4 — `EvalDiff.transitions` taxonomy-blind  ✅ RESOLVED (documentation + optional tag)

**Was:** `transitions: tuple[tuple[str, str, str], ...]` with no
indication which taxonomy produced the regime-name strings inside.

**Now:** Two changes:
1. `EvalDiff.taxonomy_name: str = ""` field added with default. The
   action-space path populates it (`gates.eval_diff` reads
   `getattr(taxonomy, "name", "")` and stamps it on the returned
   `EvalDiff`). Empty string preserves direct-caller behavior in
   tests where `EvalDiff` is constructed inline.
2. The `transitions` field's docstring now states the regime-name
   strings are TAXONOMY-LOCAL and gives `taxonomy_name` as the
   taxonomy-of-origin tag.

`Attribution` got the same `taxonomy_name` field for symmetry.

**Fix locations:**
- `src/regimes/loop/gates.py:241-265` — `EvalDiff.taxonomy_name` +
  docstring update.
- `src/regimes/loop/gates.py:364-365` — `eval_diff` stamps the
  taxonomy name onto the result.
- `src/regimes/loop/attribute.py:39-45` — `Attribution.taxonomy_name`.
- `src/regimes/loop/attribute.py:97-99` — `attribute()` stamps it.

---

## Summary table — after Phase 1.5

| Leak | Status | Severity | Pre-1.5 workaround | Post-1.5 |
|---|---|---|---|---|
| 1 — `attribute.py` LME classify | resolved | low | none (attribution labels were "unclassified") | `attribute(..., taxonomy=lctx.target.taxonomy)` |
| 2 — `gates.eval_diff` LME classify/histogram | resolved | **high** | `SqlActionSpace.eval_diff` re-implemented body | one-line delegation to `gates.eval_diff(taxonomy=..., install=..., revert=...)` |
| 3 — `sandbox_gate` LME call shape + float coercion | resolved | medium | `SqlActionSpace.sandbox_gate` re-implemented body | one-line delegation to `gates.sandbox_gate(call_fn=..., value_validator=...)` |
| 4 — `EvalDiff.transitions` taxonomy-blind | resolved | low | none | optional `taxonomy_name` field + docstring |

## Verification (run on commit of this branch)

- **`pytest -q`**: 189 passed, 3 failed (the same three pre-existing
  environmental `tests/test_split.py` failures as Phase 2).
- **`python scripts/run_loop.py --mode mock --full`**: report.json is
  byte-equal to the Phase 2 baseline. LongMemEval behavior is
  unchanged.
- **`python scripts/run_sql_loop.py --mode mock --full`**: identical
  event sequence (110 events), identical baseline (57.9%), identical
  4 promotions, identical stop reason. Only the
  `attribution.recorded.transitions` payloads differ — they now use
  the real SQL regime names instead of `"unclassified"`. **This was
  leak 1's whole purpose.**
- **Delegation evidence** (`git diff --stat`):
  - `src/regimes/targets/sql/action_space.py`: 166 lines deleted, 56
    added — net ~halved. `eval_diff` and `sandbox_gate`
    re-implementations gone.

## New leak discovered while fixing these

One small structural note, **not blocking**:

- `regimes.target.py` re-exports `EvalDiff`, `SandboxResult`,
  `StaticResult`, `PromotionDecision` from `regimes.loop.gates`. To
  avoid a circular import (`gates → target → gates`), both
  `gates.py` and `attribute.py` reference a local duck-typed
  `_TaxonomyLike` instead of `regimes.target.RegimeTaxonomy`. It's
  structurally identical to the protocol in `regimes.target` but it
  IS a duplicate. The clean fix is to move the four gate-result
  dataclasses out of `gates.py` into `regimes.target` so the
  inheritance goes the other way; that's bigger churn than this
  branch covers and the duck-typing works fine. Filing as a separate
  cleanup-ticket level item.
