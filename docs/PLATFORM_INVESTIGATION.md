# Platform Investigation: Generalizing the Regimes Loop to Multiple Targets

Goal: turn this repo from a LongMemEval-specific eval-improvement loop into a
general loop hosted on ActiveGraph that can diagnose and improve any
ActiveGraph-based target system. LongMemEval is target #1; a text-to-SQL
ActiveGraph agent is target #2.

This is a map only. Nothing is refactored.

---

## 1. Generalization boundary

Module-by-module classification of what is target-agnostic vs. baked-in to
LongMemEval / score-transforms.

### `src/regimes/loop/runner.py` — **target-agnostic**

Pure orchestration: seeds `loop.start`, drains the runtime, collects terminal
events into a `LoopReport`. The `LoopContext` it sets up (`runner.py:72-80`)
holds only generic Python objects (`eval_backend`, `author`, `instances`,
`confirm_instances`). No retrieval vocabulary anywhere. The `LoopReport`
fields (`runner.py:35-47`) — `histogram`, `baseline`, `promotions`,
`discards`, `attributions`, `transform_log` — are generic.

Touchpoint to LongMemEval: none.

### `src/regimes/loop/behaviors.py` — **mostly agnostic; one LME-shaped helper**

The nine `@behavior` functions
(`behaviors.py:177,208,245,301,333,377,410,481,507,569`) drive the chain on
generic event types. The only LongMemEval-specific code is
`_outcome_summary()` at `behaviors.py:46-108`, which reads retrieval-shaped
fields off `Outcome`:

- `o.gold_evidence_turn_ids`, `o.has_evidence_turn_ids()` (`behaviors.py:79,82,85`)
- `o.evidence_rank_positions()`, `o.evidence_ranked_top_k()`,
  `o.evidence_selected()`, `o.evidence_dropped_at_budget()`,
  `o.evidence_in_scores()`, `o.evidence_max_score()`
  (`behaviors.py:79-105`)
- `o.selected_turn_ids`, `o.truncated`, `o.score_error`, `o.scores`
  (`behaviors.py:96,97,101`)

Everything else — the static/sandbox/eval-diff/promote/attribute chain —
operates on whatever `Outcome` carries, via methods on `EvalResult` that
ARE generic (`overall_accuracy`, `per_type_accuracy`).

Two other LME-leaning details:

- `_choose_target()` (`behaviors.py:644-657`) reads from `regimes.loop.regimes.REGIMES`
  — the regime taxonomy is LME-specific (see §regimes.py below), but the
  function itself only knows about `optimizable + seam_reachable` flags.
- `_name_wall()` (`behaviors.py:617-641`) hard-codes recommendation strings
  for the four retrieval regimes (`retrieval-signal-gap`, `assemble-internal`,
  `scoring-error`, with a generic fallback). LongMemEval-specific.

### `src/regimes/loop/gates.py` — **mostly agnostic; two LME couplings**

The four gates are pure functions on their inputs. The only fields they pull
off `EvalResult` are `overall_accuracy()`, `per_type_accuracy()`, and the
per-question `o.correct` / `classify(o).name` (used by `_per_question_regime`
at `gates.py:236-246`, and by `_regime_counts` at `gates.py:231-233`).

LME-shaped pieces:

- **Sandbox-gate probe contract** (`gates.py:340-347` in behaviors.py /
  `gates.py:156-205` body). Probes are built from
  `{"scores": dict(o.scores), "question": "", "question_date": ""}` —
  the score-transform signature. The gate asserts the transform returns a
  dict whose keys are a subset of the input scores. This is the
  score-transform action space hard-wired into the gate. A SQL target's
  action space (e.g. "edit the prompt-assembly behavior") will not pass
  through this gate as-is.
- **Static-gate signature pin**: `REQUIRED_SIGNATURE_PARAMS = ("scores", "graph",
  "question", "question_date")` at `gates.py:58`. The whitelist
  `IMPORT_WHITELIST = frozenset({"math"})` at `gates.py:40` and the
  required-fn-name `"transform"` (`gates.py:68,131`) are also score-transform
  vocabulary.
- **Promotion floor** (`gates.py:307-329`): `multi-session` is referenced by
  name as a per-type floor at `gates.py:324` — that's an LongMemEval question
  type. Other targets won't have it.

### `src/regimes/loop/attribute.py` — **target-agnostic**

`attribute(before, after)` (`attribute.py:48-68`) joins per-qid regime
classifications between two `EvalResult`s. Uses `o.question_id` and
`o.correct`; everything else is delegated to `classify()`. Nothing
retrieval-shaped.

### `src/regimes/loop/regimes.py` — **fully LongMemEval-specific**

The whole file is the LongMemEval action-space taxonomy. Every detector
(`regimes.py:131-275`) and every helper (`_gold_sids`, `_gold_in_scores`,
`_gold_dropped_at_budget`, `_well_ranked_gold_coverage`) reads retrieval
fields off `Outcome`:

- `o.answer_session_ids`, `o.scores`, `o.ranked`, `o.selected_turn_ids`,
  `o.decisions`, `o.truncated`, `o.score_error`,
  `o.gold_evidence_turn_ids`
- the helpers turn turn-id strings via `tid.split("#", 1)[0]` to recover
  session ids (LME's `{session_id}#{turn_idx}` shape) — `regimes.py:106,123,156,...`

The six built-in regimes (`scoring-error`, `budget-truncation`,
`assembly-crowding`, `retrieval-signal-gap`, `assemble-internal`,
`unclassified`) and `PRIORITY` (`regimes.py:376-383`) are all LME-shaped.

`register_regime()` (`regimes.py:399-426`) is the extension point — a new
target would either replace `REGIMES`/`PRIORITY` outright or register
target-specific regimes there.

### `src/regimes/eval/types.py` — **LongMemEval-specific Outcome shape**

`Outcome` (`types.py:33-197`) is explicitly a retrieval outcome. All evidence
helpers and the docstring at `types.py:1-20` document this: "Diagnose has to
be able to classify failures into regimes without re-running the agent".
Specifically retrieval-shaped fields:

- `answer_session_ids` (`types.py:39`)
- `selected_turn_ids`, `n_seeds`, `n_expanded`, `truncated`, `running_tokens`,
  `decisions`, `scores`, `ranked`, `applied_transforms`
  (`types.py:50-58`)
- `gold_evidence_turn_ids` (`types.py:79`)
- all the `gold_*`, `evidence_*` derived helpers
  (`types.py:83-181`)

Generic / cross-target fields:

- `question_id`, `question_type`, `correct`, `judge_label`, `judge_raw`,
  `hypothesis`, `error`, `score_error`, `run_id`, `is_abstention`
  (`types.py:36-67`)

`EvalResult` (`types.py:206-237`) is target-agnostic: outcomes list, aggregate
dict, backend tag, run_dir, config — `overall_accuracy()` and
`per_type_accuracy()` are computed off the outcomes-side fields and would
work for any target that fills `correct` + `question_type`.

The `Reader` and `Judge` protocols (`types.py:246-275`) bake in LME-shaped
signatures: `Reader.answer(context, question, question_id)` and
`Judge.judge(hypotheses_path, references_path, run_dir)`. A SQL target's
"answer" is a SQL string; the "judge" is a result-set comparator. These
protocols are LME-specific but at least narrow: each target can have its own
Reader/Judge analog without disturbing `Outcome`.

### `src/regimes/eval/real.py` — **fully LongMemEval-specific**

The wrapper that bridges `RealEval` to the LongMemEval upstream harness:
shells to `evaluate_qa.py`, parses LME-format results files, builds Outcomes
from agent traces + LME judgments, walks `instance["haystack_session_ids"]`
etc. `RealEval.run_on_split` (`real.py:450-616`) is conceptually
target-specific — for each target we'll want its own `RealEval`-shaped
backend.

### `src/regimes/agent/` — **score-transform-pipeline-specific**

The four-behavior chain (`agent/behaviors.py`):
`question.asked → turns.scored → turns.transformed → turns.expanded →
context.assembled`. This is a specific retrieval architecture. The
`transforms.py` pipeline (`transforms.py:42-110`) is the seam: a global
list of `(name, callable)` entries the `agent.transform_pipeline` behavior
walks on every `turns.scored` event. Promote = append; discard = revert.

The contract `transform(scores: dict, graph, question, question_date) -> dict`
(`transforms.py:32`) is the action space. The loop's static gate and the
sandbox gate's probe construction both encode it. Reusable for any score-
based system but not for, e.g., "rewrite a prompt template" or "swap a
schema-linking step".

---

## 2. Proposed Target interface (sketch only)

The minimum a target must provide to plug into the loop:

```python
# src/regimes/target.py (proposed)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Callable

from regimes.eval.types import EvalResult


@runtime_checkable
class EvalBackend(Protocol):
    """What the loop calls during baseline / eval-diff / attribute /
    confirm. Already the de-facto contract — both RealEval and MockEval
    implement it; the loop's behaviors.py only ever calls run_on_split."""
    def run_on_split(self, instances: list[Any]) -> EvalResult: ...


@dataclass(frozen=True)
class DraftedChange:
    """Generalizes loop.hypothesize.DraftedTransform. The CONTENT of
    `source` and the action-space-specific decisions about compile /
    install / revert live in the target's ActionSpace."""
    name: str
    source: str               # human-readable; may be Python, JSON, a prompt fragment
    target_regime: str
    author: str               # "stub" | "claude-..." | etc.
    rationale: str = ""


@runtime_checkable
class ActionSpace(Protocol):
    """Encapsulates draft → static-validate → sandbox-probe → install /
    revert for ONE class of change. For LongMemEval today this is the
    score-transform pipeline; for SQL it'd be a prompt-edit pipeline."""

    def draft(self, *, dominant_regime: str,
              failures: list[Any]) -> DraftedChange: ...
    def static_gate(self, source: str) -> "StaticResult": ...
    def compile(self, source: str) -> Any: ...
    def sandbox_gate(self, fn: Any, *, probes: list[dict]) -> "SandboxResult": ...
    def install(self, name: str, fn: Any) -> None: ...   # was transforms.promote
    def revert(self, name: str) -> None: ...
    def build_probes(self, baseline: EvalResult) -> list[dict]: ...


@runtime_checkable
class RegimeTaxonomy(Protocol):
    """Per-target detectors. Today this is the global module-level
    registry in loop/regimes.py. Generalized, each target owns its own
    detector set."""
    def classify(self, outcome: Any) -> "Regime": ...
    def histogram(self, outcomes: list[Any]) -> list["HistogramRow"]: ...
    def is_seam_reachable(self, name: str) -> bool: ...
    def REGIMES(self) -> dict[str, "Regime"]: ...
    def name_wall(self, counts: dict[str, int]) -> str: ...  # absorbs _name_wall


@runtime_checkable
class Target(Protocol):
    """The whole plugin surface. The loop calls these and nothing else."""
    name: str                            # "longmemeval" | "text-to-sql" | ...
    eval_backend: EvalBackend
    action_space: ActionSpace
    taxonomy: RegimeTaxonomy
    # outcome_summary(outcome) is the per-target replacement for
    # behaviors._outcome_summary; rest of the loop doesn't care what's
    # inside the dict it gets back.
    def outcome_summary(self, outcome: Any) -> dict[str, Any]: ...
```

### What already nearly implements this for LongMemEval

| Proposed piece | Existing code (would be re-homed) |
|---|---|
| `EvalBackend.run_on_split` | `RealEval.run_on_split` (`eval/real.py:450`) + `MockEval.run_on_split` (`loop/mock_eval.py:88`). Both already match; only `RealEval` needs the `run_dir` argument hoisted out (today the CLI wraps it — `scripts/run_loop.py:154-159`). |
| `DraftedChange` | `DraftedTransform` (`loop/hypothesize.py:38-44`). Rename + drop the score-transform-specific phrasing in docstring. |
| `ActionSpace.draft` | `StubAuthor`, `LLMAuthor` (`loop/hypothesize.py:110-221`). |
| `ActionSpace.static_gate` | `static_gate()` (`loop/gates.py:68-128`) + `REQUIRED_SIGNATURE_PARAMS`, `IMPORT_WHITELIST` — all become per-target settings. |
| `ActionSpace.compile` | `compile_transform()` (`loop/gates.py:131-140`). |
| `ActionSpace.sandbox_gate` | `sandbox_gate()` (`loop/gates.py:156-205`). |
| `ActionSpace.install / revert` | `regimes.agent.transforms.promote / revert` (`agent/transforms.py:51-62`). |
| `ActionSpace.build_probes` | The probe-construction loop in `behavior_sandbox_gate` (`loop/behaviors.py:340-346`). |
| `RegimeTaxonomy.classify, histogram, REGIMES` | The module-level functions in `loop/regimes.py:393-497`. Moves from a process-global registry to a per-target instance. |
| `RegimeTaxonomy.name_wall` | `_name_wall()` (`loop/behaviors.py:617-641`). The recommendation strings move with it. |
| `Target.outcome_summary` | `_outcome_summary()` (`loop/behaviors.py:46-108`). |

`Outcome` itself is the open question. Two options:

1. **Keep `Outcome` as the LongMemEval shape; let each target define its
   own.** The loop only touches `Outcome` through (a) `classify(o)`
   (delegated to `RegimeTaxonomy`), (b) `o.correct` / `o.question_id` /
   `o.question_type`, (c) the per-target `outcome_summary` (already
   delegated). Everything else (`o.gold_evidence_turn_ids`,
   `o.evidence_*`, `o.scores`, `o.ranked`, ...) is only read by the
   target's own taxonomy and summary code.
2. **Generalize `Outcome` to a minimal core dataclass** + a free-form
   `target_data: dict`. More disruption; weaker invariants on
   `EvalResult.outcomes`.

Option 1 reads easier — the loop's per-outcome touch points are already
narrow, so subclass-by-convention works. The contract becomes "Outcome
must have `question_id: str`, `question_type: str`, `correct: bool`".

### Loop-side changes implied

- `LoopContext` (`loop/behaviors.py:118-148`) grows a `target: Target` field;
  it stops carrying `eval_backend` / `author` directly (they're
  `target.eval_backend` / `target.action_space`).
- `behaviors.py`'s direct imports of `regimes.agent.transforms` (`behaviors.py:31,447,454,456,457` — promote/revert) become `target.action_space.install / revert`.
- `behaviors.py`'s imports from `regimes.loop.regimes` and
  `regimes.loop.gates` route through `target.taxonomy` / `target.action_space`.
- `runner.run_loop(...)` takes `target: Target` instead of (or in addition
  to) `eval_backend` and `author`.

---

## 3. How the SQL target would fit

A text-to-SQL ActiveGraph agent: ingest schema + question as events,
behaviors assemble a prompt, an LLM writes SQL, execute against sqlite,
check result.

### Reuse vs. rebuild

| Loop machinery | Reuse for SQL? |
|---|---|
| `runner.run_loop` | Yes, unchanged. |
| `loop/behaviors.py` chain | Yes — once the LongMemEval-shaped bits (`_outcome_summary`, `_name_wall`, the score-transform probe construction in `behavior_sandbox_gate`, the direct `regimes.agent.transforms` imports) are routed through `Target`. |
| `loop/gates.py` static / sandbox / eval-diff / promotion | Yes — static and sandbox become parameterized (signature, whitelist, probe shape) and live in the `ActionSpace`. Eval-diff and promotion already work on generic `EvalResult` / per-qid regime counts. |
| `loop/attribute.py` | Yes, unchanged. |
| `loop/hypothesize.py` (StubAuthor, LLMAuthor) | Partially: the **prompt-build** for the LLM is action-space-specific (it talks about score-transforms today, `loop/hypothesize.py:265-291`). The author scaffolding (model selection, code-block extraction) is reusable. |
| `agent/` four-behavior chain | **Not reused.** The SQL agent has a different behavior chain: e.g. `question.asked → schema.selected → prompt.assembled → sql.generated → sql.executed → answer.recorded`. Same ActiveGraph runtime, same `@behavior` / `Runtime` / `graph.emit` patterns — different content. |
| `agent/transforms.py` pipeline | **Not reused as-is.** The SQL target's seam isn't "post-score reweight". Its analog is "post-prompt-assembly edit step" or "schema-linking filter". Same idea (a single-channel pipeline behavior the loop's promote/revert manipulates) but a different signature and a different installed behavior. |

### Where the score-transform seam analog lives for SQL

The loop's seam is wherever `ActionSpace.install` writes and a fixed agent
behavior reads. For LongMemEval today that's `regimes.agent.transforms`
(written by `transforms.promote`, read by `behavior_transform_pipeline` in
`agent/behaviors.py:108-143`).

For SQL the most natural analog: a `regimes.sql_agent.prompt_transforms`
module with the same shape — a global ordered list of
`(name, prompt_edit_callable)` entries that a `sql_agent.prompt_pipeline`
behavior walks on every `prompt.assembled` (or `prompt.proposed`) event,
emitting `prompt.transformed`. A "prompt-edit" is e.g.

```python
def transform(prompt: str, schema: dict, question: str) -> str: ...
```

or, if we want more leverage, an action over the assembled context object
rather than a raw string. The static gate's whitelist + signature pin
parameterize cleanly: the SQL action space picks its own
`REQUIRED_SIGNATURE_PARAMS` and import whitelist. The sandbox gate's probes
become `{"prompt": str, "schema": dict, "question": str}` instead of
`{"scores": dict, ...}`.

The SQL action space is "LLM authors a transform to the prompt/assembly
step"; promoted iff held-out SQL exact-match (or execution-match) improves.
That maps directly onto the same draft → static → sandbox → eval-diff →
promote → confirm → attribute path the LME loop already runs.

### SQL target's regime taxonomy (sketch)

Failure modes the loop would want detectors for, all addressable from a
SQL Outcome carrying `(question, gold_sql, predicted_sql, exec_result,
exec_error, schema_subset_used, prompt_token_count, judge_label)`:

- `schema-link-miss`: gold table/column not in `schema_subset_used`. Optimizable + seam-reachable (prompt-edit can pull in more schema).
- `prompt-truncation`: schema overflowed the prompt budget; analogue of `budget-truncation`.
- `sql-syntax-error`: `exec_error` is a parse error. Optimizable via prompt edits ("use SQLite syntax").
- `sql-runtime-error`: parse OK but exec raises (no such column, wrong join).
- `sql-correct-shape-wrong-row`: query ran, result set is wrong shape / wrong rows. Often reader-side / model-internal — analog of `assemble-internal`.
- `unclassified`.

This taxonomy is a sibling to LongMemEval's. Same `Regime` dataclass, same
priority discipline, different detectors.

---

## 4. SQL task data

### What's in the repo today

No SQL dataset, no sqlite executor, no Spider mirror — grep for
`sql|sqlite|spider` over `src/`, `tests/`, `scripts/` returns nothing.

### Lightest path: committed synthetic SQL fixture

The README documents that HuggingFace is unreachable in this build env
(`README.md:69-83`), which is why LongMemEval ships with
`fixtures/synthetic_lme.json` — a 200-question deterministic fixture
generated by `scripts/build_fixture.py`, byte-identical on re-run.

Mirror that exactly for SQL. Concrete plan:

1. **A tiny hand-built schema fixture**: 2–3 SQLite databases as inline DDL
   + sample rows, committed as `fixtures/synthetic_sql.json` (or a
   small `.sql` per db plus a JSON index). Example domains: a 3-table
   `library`, a 4-table `flights`, a 2-table `inventory`. ~30–50 questions
   total, each with `{question, db_id, gold_sql, expected_result_set}`.
2. **`scripts/build_sql_fixture.py`**: deterministic generator (seed=42)
   that produces both the schemas and the question/SQL pairs — same
   byte-equality discipline as `scripts/build_fixture.py`.
3. **A sqlite executor in `regimes.sql_agent.exec`**: opens the db, runs
   the predicted SQL, returns `(result_set, error)`. Standard library
   only (`sqlite3`), no network.
4. **Comparison rule**: exec-match (set-of-rows equality, ignoring order
   when the gold SQL has no `ORDER BY`). This is the equivalent of LME's
   `evaluate_qa.py` for the SQL target.

A Spider dev subset would be nicer signal-wise, but per the README's
HuggingFace constraint it can't be pulled in this env. If a Spider mirror
becomes feasible later (cached locally, committed under `fixtures/`), the
same exec-match comparator works on it unchanged.

Suggested initial scope for Phase 2 v0: ~30 hand-crafted questions across
3 schemas, all answerable with single-table SELECT or simple JOIN — enough
to exercise `schema-link-miss`, `prompt-truncation`, and
`sql-syntax-error` regimes.

---

## 5. Venv / health check

The container started fresh; nothing was installed at session start. I ran
`pip install -e .` so the dependency check below reflects what the project
declares it needs vs. what's actually available.

| Check | Result |
|---|---|
| `pip show anthropic openai numpy tiktoken \| grep Version` | None of the four are installed. `pip show` prints `WARNING: Package(s) not found: anthropic, numpy, openai, tiktoken`. |
| `python -c "import activegraph; print(activegraph.__version__)"` | **1.0.5.post2** — matches `pyproject.toml`'s pin. |
| `pytest -q` final line | `3 failed, 162 passed in 0.88s` (162 pass, 3 fail) |

### Flags

- **anthropic/openai/numpy/tiktoken are absent.** This is expected for the
  default install — `pyproject.toml` puts `openai` and `anthropic` under
  the `[eval]` optional extra (lines 23-29), and `numpy`/`tiktoken` aren't
  declared at all (the agent uses a `HashEmbedder` by default with no
  numpy dependency). The "cross-repo install" you mentioned — if it had
  forced downgrades — would only have shown up if those packages had been
  installed in this venv; they weren't.
- **3 test failures, all in `tests/test_split.py`**:
  - `test_committed_split_required_types_in_both`
  - `test_loader_rejects_missing_abstention`
  - `test_split_generator_is_deterministic`

  Cause: `config/split.json` was generated against the real LongMemEval
  dataset (`source: ../activegraph-longmemeval/data/longmemeval_s_cleaned.json`,
  `n_total: 500`), but in this container the real data isn't present so
  the deterministic generator runs against `fixtures/synthetic_lme.json`
  (`n_total: 200`). The generator's output legitimately drifts from the
  committed split. These are **environmental** failures, not regressions
  — they fail any time tests run without the real LME data sitting next to
  this checkout. The 94 loop tests, the agent tests, and the eval tests
  (162 total) all pass.

- **No corruption.** Nothing in the venv reports a broken numpy / tiktoken
  state; they simply aren't installed.

---

## 6. Naming / model note — every hardcoded `claude-sonnet-4-5`

`claude-sonnet-4-5` no longer resolves at the API. The working string is
`claude-sonnet-4-6` and it should read from `BEHAVIORDRAFTS_MODEL` (and a
parallel env var for the reader). Locations to fix in the Phase 1 refactor:

| File:Line | Context |
|---|---|
| `src/regimes/eval/real.py:58` | `AnthropicReader.name: str = "claude-sonnet-4-5"` (class default; reader model). |
| `src/regimes/eval/types.py:248` | Reader-protocol docstring says "Real implementation is AnthropicReader (claude-sonnet-4-5, ...)". |
| `src/regimes/loop/hypothesize.py:43` | `DraftedTransform.author` doc-comment: `"stub" \| "claude-sonnet-4-5"`. |
| `src/regimes/loop/hypothesize.py:149` | `DEFAULT_LLM_MODEL = "claude-sonnet-4-5"` (used by both LLMAuthor's default and `build_real_author`). |
| `README.md:167` | Example shows `claude-sonnet-4-5, T=0, tool-free`. |
| (No occurrences in tests or scripts.) | |

`build_real_author` (`hypothesize.py:152-161`) already reads
`BEHAVIORDRAFTS_MODEL` from env — good. The fix is two-pronged:

1. Change every `claude-sonnet-4-5` literal (4 code sites + 1 README site)
   to `claude-sonnet-4-6`.
2. Add the same env-var pattern to `AnthropicReader` in `eval/real.py`:
   construct via a `build_real_reader(model=None)` helper that reads
   `REGIMES_READER_MODEL` (or reuses `BEHAVIORDRAFTS_MODEL`) with the
   `claude-sonnet-4-6` fallback. Drop the class-default literal entirely
   so a stale default can't sneak back in.

---

## Summary for the refactor spec

**Phase 1** can be tightly scoped:

1. Define `Target` / `EvalBackend` / `ActionSpace` / `RegimeTaxonomy`
   protocols in a new `src/regimes/target.py`.
2. Re-home LongMemEval's pieces under
   `src/regimes/targets/longmemeval/` (eval backend, action space adapter
   wrapping the existing score-transform pipeline, taxonomy adapter
   wrapping the existing regimes module, outcome summary). No behavior
   change.
3. `LoopContext` and `runner.run_loop` take a `Target`. The loop's
   behaviors stop importing `regimes.agent.transforms` / `regimes.loop.regimes`
   directly and go through `target.*`.
4. Fix the `claude-sonnet-4-5` strings, drop the class-default literal on
   `AnthropicReader`, route both author and reader models through env
   vars.
5. Existing 162 tests should pass unchanged once the LongMemEval target
   wraps the existing pieces 1:1. The 3 split tests will continue to fail
   for the same environmental reason — orthogonal to this refactor.

**Phase 2** adds `src/regimes/targets/text_to_sql/` implementing the same
`Target` protocol against a sqlite executor and the new
`fixtures/synthetic_sql.json`.
