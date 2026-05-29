# Phase 1.5 Cleanup: LME-shape leaks discovered when building the SQL target

The Phase 1 refactor (commit `b94edbe`) moved the LongMemEval couplings
behind the `Target` interface and preserved byte-equal LongMemEval
behavior. Phase 2 (the SQL target) exercised that interface with a
genuinely different action space and taxonomy and surfaced four
places in the loop-control code where LongMemEval shape still leaks
through. The SQL target works around each — but they should be fixed
before Phase 3 / before any third target.

These were discovered, not introduced. The SQL target is committed
with the workarounds in place; this doc is the punch list.

---

## Leak 1 — `loop/attribute.py` calls LongMemEval's `classify` directly

**File:** `src/regimes/loop/attribute.py:22`, `:42-44`

```python
from regimes.loop.regimes import classify   # <-- LME-pinned import
...
def _per_qid_regime(result: EvalResult) -> dict[str, str]:
    out: dict[str, str] = {}
    for o in result.outcomes:
        out[o.question_id] = "correct" if o.correct else classify(o).name
```

**Symptom for SQL:** Every SqlOutcome goes through LME's `classify` and
lands on `"unclassified"` (its detectors all return False on
SqlOutcome's empty LME-shaped fields). So `attribution.recorded` events
report transitions like `("sql_co_q01", "unclassified", "correct")`
instead of `("sql_co_q01", "schema-misunderstanding", "correct")`.

**Why it doesn't crash:** `SqlOutcome` subclasses `Outcome` so the LME
detector reads `.answer_session_ids` / `.scores` / `.gold_evidence_turn_ids`
without `AttributeError` — they're just empty.

**Fix:** Thread `target.taxonomy.classify` through `attribute()` (take
a taxonomy argument; the loop's `behavior_attribute` already has
`lctx.target.taxonomy` in scope).

---

## Leak 2 — `loop/gates.py::_per_question_regime` / `_regime_counts` call LME classify/histogram

**File:** `src/regimes/loop/gates.py:231-233`, `:236-246`

```python
from regimes.loop.regimes import classify, histogram
...
def _regime_counts(result: EvalResult) -> dict[str, int]:
    rows = histogram(result.outcomes)
    return {r.regime: r.count for r in rows}

def _per_question_regime(result: EvalResult) -> dict[str, str]:
    ...
    out[o.question_id] = classify(o).name
```

**Symptom for SQL:** Same as leak 1 — for SqlOutcomes the LME
taxonomy returns `unclassified` for every failure, so
`gates.eval_diff` would report `regime_before == regime_after`
constantly. The `target_delta` it computes would always be `0`
because the target regime ("schema-misunderstanding" etc.) doesn't
exist in LME's registry, and `r_after.get(target_regime, 0)` is
always 0. **Result: promotion_decision always rejects** ("target
regime did not shrink: target_delta=0"). Every SQL transform would
be discarded.

**Workaround in Phase 2:** `SqlActionSpace.eval_diff`
(`src/regimes/targets/sql/action_space.py:139-180`) re-implements the
same diff logic, calling `self.taxonomy.classify` / `self.taxonomy.histogram`
instead of going through `gates.eval_diff`. Verified: with the
workaround, all four stub SQL transforms get promoted because the
SQL taxonomy correctly sees the target regime shrinking.

**Fix:** Take an optional `taxonomy=` argument on `gates.eval_diff`
(default = LME's, so behavior unchanged). Then
`LongMemEvalActionSpace.eval_diff` and `SqlActionSpace.eval_diff` can
both go back to a single shared `gates.eval_diff` call.

---

## Leak 3 — `loop/gates.py::sandbox_gate` hard-codes the score-transform call shape

**File:** `src/regimes/loop/gates.py:156-205`

```python
def sandbox_gate(fn, *, probes, time_budget_s=2.0):
    ...
    for p in probes:
        input_scores = dict(p.get("scores", {}))
        out = fn(input_scores, None, p.get("question", ""),
                 p.get("question_date", ""))      # <-- LME signature pinned
        ...
        try:
            {tid: float(v) for tid, v in out.items()}   # <-- float coercion
        except (TypeError, ValueError) as e:
            reasons.append(f"non-float values at probe {n_done}: {e}")
```

The call shape `fn(scores, None, question, question_date)` and the
float-coercion-on-values check are both score-transform-specific.

**Workaround in Phase 2:** `SqlActionSpace.sandbox_gate`
(`src/regimes/targets/sql/action_space.py:73-117`) reimplements the
gate body with the prompt-transform call shape `fn(prompt_parts,
question, schema_meta)` and no float coercion (prompt_parts values are
strings/lists). It still returns the same `SandboxResult` dataclass so
the loop's `behavior_sandbox_gate` sees the documented shape.

**Fix:** Make `sandbox_gate` take a `call_fn` callable
(`call_fn(fn, probe) -> dict`) and an optional `value_validator`
(`Callable[[dict], None]` that raises on bad values). The current
LME-shaped behavior becomes one specific `(call_fn, value_validator)`
pair; the SQL action space supplies its own.

---

## Leak 4 — `regimes.target.EvalDiff.transitions` carries regime-name strings whose meaning is per-taxonomy

**File:** `src/regimes/loop/gates.py:223-224` (definition), re-exported
via `regimes.target`.

```python
transitions: tuple[tuple[str, str, str], ...] = ()
# Each row is (qid, regime_before_name, regime_after_name)
```

This isn't broken per se — it's a generic shape — but the regime names
inside it are produced by whatever `classify` made them, and (per
leaks 1 + 2) that's currently LME's `classify` for any non-SqlAction-
Space caller. So the strings inside `transitions` are
LME-namespace-shaped.

Not a leak the SQL target hits, since `SqlActionSpace.eval_diff`
produces SQL-named transitions internally. Worth noting for future
clarity: a future `Target` whose taxonomy shares any regime names
with LME's would alias them; the `transitions` field has no
taxonomy-of-origin tag.

**Fix:** Either tag transitions with a taxonomy name, or document that
`EvalDiff.transitions` is target-local. The simplest fix is to make
the leak-2 fix (taxonomy threaded through `gates.eval_diff`)
authoritative and call out the field as target-local in its docstring.

---

## Summary

| Leak | Symptom in SQL | Workaround | Severity |
|---|---|---|---|
| 1 — `attribute.py` LME classify | attribution events use `"unclassified"` labels instead of real SQL regimes | none; cosmetic | low |
| 2 — `gates.eval_diff` LME classify/histogram | would block ALL SQL promotions | `SqlActionSpace.eval_diff` re-implements diff with SQL taxonomy | **high** |
| 3 — `sandbox_gate` LME call shape + float coercion | wouldn't be callable on prompt-transform fn | `SqlActionSpace.sandbox_gate` re-implements with prompt-transform shape | medium |
| 4 — `EvalDiff.transitions` taxonomy-blind | string labels in transitions could collide across taxonomies | none | low |

Fix order: **leak 2 first** (blocks new targets from promoting at all),
then leak 3 (forces every new ActionSpace to re-implement
`sandbox_gate`), then leak 1 (cosmetic but cheap once leak 2 is done),
then leak 4 (documentation).
