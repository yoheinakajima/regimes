# Regimes — project status

A checkpoint document for an eventual paper, grounding what the system is, what
the artifacts say, and what is still unverified. Every claim is tied to a file
path + line range in the repo or to a tagged artifact. Numbers the author
reports from a real-mode run on their machine are marked **[user-reported,
not in-repo]** because `runs/` is gitignored (`.gitignore:14`) and only mock-mode
artifacts are reproducible inside this environment.

---

## 1. Thesis

Regimes is an autonomous eval-improvement loop that runs natively on the
published `activegraph` runtime (`pyproject.toml:11` pins `activegraph==1.0.5.post2`).
Every phase of the loop — baseline, diagnose, hypothesize, the four lifecycle
gates, attribute, stop — is a real `@behavior` registered with the runtime and
emits real events into the runtime's event log
(`src/regimes/loop/behaviors.py:170-602`; phase event vocabulary in
`src/regimes/loop/events.py:34-49`). The loop diagnoses why a graph-based
retrieval agent fails on LongMemEval, attempts to fix the seam-reachable
failures via gated score-transforms injected at `agent.transform_pipeline`
(`src/regimes/agent/transforms.py:1-23`), and stops + names the wall
(`src/regimes/loop/behaviors.py:605-629`) when the remaining failures are
outside its action space.

The dogfooding is structural: the agent under test is itself a four-behavior
chain on the same runtime (`README.md:131-136`,
`src/regimes/agent/behaviors.py`), and the loop's audit trail and the agent's
audit trail are the same kind of event log. ActiveGraph is optimizing an
ActiveGraph system.

---

## 2. Architecture

### 2.1 Phase event chain

Single seed `loop.start` → runtime drains the chain → terminal `loop.stopped`.
The sequence below is the contract documented at `src/regimes/loop/events.py:7-29`
and implemented as one `@behavior` per phase in `src/regimes/loop/behaviors.py`:

```
loop.start                                            (seed, runner-emitted)
  → behavior_run_baseline                            (behaviors.py:170)
baseline.recorded
  → behavior_diagnose                                (behaviors.py:201)
regime.histogram                                     [first emitted artifact]
  → behavior_hypothesize                             (behaviors.py:238)
transform.drafted                                    (inert — name + source)
  → behavior_static_gate                             (behaviors.py:292)
transform.static_passed | transform.static_rejected
  → behavior_sandbox_gate                            (behaviors.py:323)
transform.sandbox_passed | transform.sandbox_rejected
  → behavior_eval_diff                               (behaviors.py:366)
transform.eval_diff                                  (per-type + overall deltas)
  → behavior_promote                                 (behaviors.py:399)
transform.promoted | transform.discarded
  → behavior_attribute                               (behaviors.py:468, promote only)
attribution.recorded
  → behavior_iterate_after_{promote,discard}         (behaviors.py:494, 501)
loop.iterate | loop.stopped                           (terminal)
```

`run_loop()` does only three Python things outside the runtime: build
`Graph(ids=IDGen(), clock=FrozenClock(...))`, seed `loop.start`, then
post-extract terminal payloads from `graph.events` into a `LoopReport`
(`src/regimes/loop/runner.py:50-138`). The audit trail IS the event log;
the convenience fields (`baseline`, `histogram`, `promotions`, `discards`,
`attributions`, `stopped`, `transform_log`) are pre-plucked.

### 2.2 Action space — score-transforms at the `turns.scored` seam

The only thing the loop is allowed to change is one ordered pipeline of
score-transforms injected between the scoring step and assembly
(`src/regimes/agent/transforms.py:1-55`). The signature is fixed:

```python
def transform(scores: dict[str, float], graph,
              question: str, question_date: str) -> dict[str, float]
```

A single `agent.transform_pipeline` behavior fires on every `turns.scored` event
and walks the pipeline left-to-right (`transforms.py:18-22`). Promotion calls
`transforms.promote(name, fn)`; the loop never invokes a transform any other
way. The static-analysis gate (`src/regimes/loop/gates.py:68-128`) enforces:
exact signature pin, math-only import whitelist (`IMPORT_WHITELIST = {"math"}`,
`gates.py:40`), banned dunders/builtins (`gates.py:42-56`), AST = imports +
one `def` (no top-level statements). That AST whitelist is the entire safety
model — there is no runtime sandbox beyond the time-budgeted probe gate
(`gates.py:156-205`).

### 2.3 Authorship without authority

`hypothesize` consumes the histogram and produces a `DraftedTransform`
(`src/regimes/loop/hypothesize.py:38-45`). The author proposes a name + source
string; the loop decides whether to promote based on the gate outcomes. Two
authors are provided:

- `StubAuthor` (`hypothesize.py:111-141`): deterministic, picks from a tiny
  pre-written library (`_STUB_LIBRARY`, `hypothesize.py:65-103`) keyed by
  target regime. Used for every test in this container.
- `LLMAuthor` (`hypothesize.py:150-223`): calls Claude
  (`claude-sonnet-4-5`, T=0.2, 1024 max tokens) with the failing outcomes
  + targeted regime + signature hint, extracts the function body from the
  reply. Construction asserts `ANTHROPIC_API_KEY` + `anthropic` import or
  raises `ConfigurationError` (caller-fixable, `hypothesize.py:159-170`).

Both return the same `DraftedTransform` dataclass. The loop carries the source
string only — never callables — through the event payloads; the static gate
compiles it in-place before the sandbox runs (`gates.py:131-140`).

### 2.4 Held-out discipline

The loop refuses to start without `config/split.json`, a frozen
OPTIMIZE/CONFIRM partition (`src/regimes/split.py:67-113`). The loader raises
`activegraph.ConfigurationError` on missing file, empty sets, or
`OPTIMIZE ∩ CONFIRM ≠ ∅` (`split.py:95-101`). The committed split is OPTIMIZE-50
+ CONFIRM-100, seed=42, source `longmemeval_s_cleaned.json`
sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
(`config/split.json:1-7`).

CONFIRM is touched only inside `behavior_promote` and only when a transform has
already passed all four gates on OPTIMIZE
(`src/regimes/loop/behaviors.py:438-449`). The promote behavior runs
CONFIRM-with-transform vs CONFIRM-without-transform exactly once per
promotion and records the delta in the `transform.promoted` payload's
`confirm_delta`. Every transform attempt (passed / rejected at any gate /
promoted / discarded) is appended to `LoopReport.transform_log`
(`behaviors.py:648-666`) so the headline CONFIRM number is reportable as
best-of-N rather than a winner-take-all selection.

---

## 3. Regime taxonomy + detectors

Taxonomy and detectors live in `src/regimes/loop/regimes.py`. Each regime is a
`Regime(name, detector, optimizable, seam_reachable, description)` dataclass
(`regimes.py:75-86`). Detectors are pure functions of one `Outcome` — no
graph re-walk, no I/O — so they're trivially replayable
(`regimes.py:1-26`).

| Regime | optimizable | seam-reachable | Detector |
|---|---|---|---|
| `scoring-error` | no | no | `o.score_error` set, OR gold turns entirely absent from `o.scores` dict (`regimes.py:131-151`) |
| `retrieval-signal-gap` | no | no | gold scored at all, but no gold **evidence** turn appears in top-`WELL_RANKED_K` of `o.ranked` (`regimes.py:258-275`) |
| `budget-truncation` | **yes** | **yes** | `o.truncated` AND gold evidence ranked well AND gold evidence appears in `o.decisions` with `included=False, reason='budget'` (`regimes.py:209-235`) |
| `assembly-crowding` | **yes** | **yes** | gold well-ranked but evidence coverage < `ASSEMBLE_COVERAGE_FLOOR` (0.5) of well-ranked gold in `selected_turn_ids` (`regimes.py:238-255`) |
| `assemble-internal` | no | no | well-ranked gold AND coverage ≥ 0.5 AND answer still wrong — reader/format/judge issue (`regimes.py:186-206`) |
| `unclassified` | no | no | catch-all (`regimes.py:278-281`) |

Knobs: `WELL_RANKED_K = 20`, `ASSEMBLE_COVERAGE_FLOOR = 0.5`
(`regimes.py:63-64`). Earlier values (`TOP_K=5`) were too narrow — see Bug D
in §5. Priority order (`regimes.py:376-383`):

```
scoring-error → retrieval-signal-gap → budget-truncation
              → assembly-crowding → assemble-internal → unclassified
```

`classify()` short-circuits at the first match (`regimes.py:440-450`).
`histogram()` returns one `HistogramRow` per regime in PRIORITY order with the
per-failure qid list (`regimes.py:469-497`).

### 3.1 Evidence-turn granularity — the structural fix

This is the single most important detail in the taxonomy. Each detector
reasons at **evidence-turn granularity** when the Outcome carries
`gold_evidence_turn_ids`, falling back to **session-level** only when the
dataset doesn't mark evidence per turn (e.g. the synthetic fixture).

Evidence-turn IDs come from one of two LongMemEval encodings, extracted in
`_extract_evidence_turn_ids()` (`src/regimes/eval/real.py:369-416`):

1. Per-turn `has_answer: true` flag on individual haystack turns (the
   standard `longmemeval_s` format).
2. Top-level `answer_evidences` list of `{session_id, turn_idx}` (older /
   oracle variants).

Both produce ids in the agent's `{session_id}#{turn_idx}` shape, matching
`selected_turn_ids`, `ranked`, and `scores`.

**Why this matters.** In LongMemEval the gold session usually has multiple
turns, only a subset of which are evidence. The agent's seed phase reliably
picks at least one high-scoring **non-evidence** turn from the gold session.
Session-level detectors then read "a gold turn was retrieved / selected /
dropped at budget" and misclassify:

- `eac54add` (real LME): the only evidence turn ranked **126**. The gold
  session also contained non-evidence filler ranked **2, 3, 11**. Session-level
  reasoning saw "gold in top-K" → not signal-gap; saw filler dropped at the
  budget wall → falsely budget-truncation
  (`tests/test_regime_recalibration.py:344-358`, the
  `test_eac54add_does_not_misclassify_as_budget_truncation` regression test).

The evidence-level helpers on `Outcome`
(`src/regimes/eval/types.py:120-181`: `evidence_ranked_top_k`,
`evidence_selected`, `evidence_dropped_at_budget`, `evidence_rank_positions`,
`evidence_in_scores`, `evidence_max_score`) are what the detectors actually
call now.

### 3.2 The never-retrieved structural invariant

`detect_budget_truncation` was tightened to REQUIRE non-empty
`evidence_ranked_top_k(WELL_RANKED_K)` before it even looks at the budget-drop
trail (`regimes.py:229-234`). The contract is structural: a never-retrieved
(poorly-ranked) evidence turn **cannot** be classified as budget-truncation,
even if a high-scoring filler turn from the same session shows a budget drop
in `decisions`. The priority order is defense-in-depth: signal-gap fires first
(`regimes.py:359-365`), but even if a future detector loosening accidentally
fires budget-truncation on a never-retrieved case, the well-ranked-evidence
precondition stops it.

The Outcome record persisted in the event log carries the evidence-level
signals so each label is auditable against its basis: `_outcome_summary()`
(`src/regimes/loop/behaviors.py:46-108`) writes `gold_evidence_turn_ids`,
`evidence_rank_positions`, `evidence_in_scores`, `evidence_well_ranked`,
`evidence_selected`, `evidence_dropped_at_budget`, `evidence_coverage`,
`well_ranked_k` into every `baseline.recorded` payload.

---

## 4. Empirical results

**⚠ Verification scope.** The numbers in this section come from a real-mode
run the author executed on their machine. `runs/` is gitignored
(`.gitignore:14`), so the report.json and the `sub_*/` subdirs are not in this
repo and I cannot independently verify the per-question signals here. The
in-container reproducible artifact is a **mock-mode** run of the same loop
(`runs/loop_001/report.json` after `python scripts/run_loop.py --mode mock
--full`); its 7-instance synthetic fixture exercises every code path but
produces different numbers, captured in §4.4 below.

### 4.1 Baseline accuracy (user-reported, real-mode)

- **Overall: 0.78** on OPTIMIZE-50.
  - Reader non-determinism observed at roughly ±2 points across runs; earlier
    runs showed 0.76. The reader (`claude-sonnet-4-5`, T=0) is the only
    stochastic element in the chain — see §7.
- **Per-type accuracy** (user-reported):
  - `single-session-user`: 1.00
  - `single-session-assistant`: 1.00
  - `knowledge-update`: 1.00
  - `multi-session`: 0.85
  - `single-session-preference`: 0.67
  - `temporal-reasoning`: 0.36–0.43 (variance across runs)

**[user-reported, not in-repo]** I have not seen the source report.json so
I cannot pin the exact counts that produce these ratios.

### 4.2 Comparison target

The comparison reference is `rag-dense-turn` at **0.90 overall / 0.71
temporal / ~2400 mean context tokens** on LME-s, n=50. The README claims this
number lives in `../activegraph-longmemeval/paper/results_tables.md`
(`README.md:316-320`). **The sibling repo is not present in this environment**
— `ls ../` shows only `regimes`. Treat the comparison number as user-reported
until that path is verifiable.

### 4.3 Failure regime distribution (user-reported, n=11 failures)

| Regime | Count | Optimizable | Seam-reachable |
|---|---|---|---|
| `budget-truncation` | 4 | yes | yes |
| `retrieval-signal-gap` | 1 | no | no |
| `assemble-internal` | 6 | no | no |

Implication: **4 of 11** failures are inside the score-transform action space.
The remaining **7** name the wall (1 signal change, 6 assemble() internals).

### 4.4 Hand-verified ground-truth cases (regression fixtures)

These three cases are pinned in `tests/test_regime_recalibration.py:141-201`
as `CaseSpec` records and parameterized into
`test_pinned_case_classifies_correctly`. The numbers below are read directly
from the test fixture (so verifiable in-repo); the labels are the regimes the
detectors must assign.

| qid | question_type | evidence ranks | non-evidence gold ranks | selected positions | expected regime |
|---|---|---|---|---|---|
| `gpt4_a1b77f9c` | multi-session | 4, 5, 6, 8, 12, 16 of ~498 | none | (4,) — 1 of 6 evidence selected | `assembly-crowding` |
| `eac54add` | temporal-reasoning | 126 of 476 | 2, 3, 11 | (2, 3) — both non-evidence | `retrieval-signal-gap` |
| `b46e15ed` | multi-session | 2, 3, 11, 166 of 480 | none | (2,) — 1 of 4 evidence selected | `assembly-crowding` |

Two notes:

1. In the user's narrative the three cases were described as
   gpt4_a1b77f9c → budget-truncation, eac54add → retrieval-signal-gap,
   b46e15ed → budget-truncation. The pinned test fixture in
   `tests/test_regime_recalibration.py:155, 199` codes the first and third as
   **assembly-crowding**, not budget-truncation. The two regimes share the
   same actionable property (both optimizable + seam-reachable) — the
   difference is whether the agent's `decisions` log records an explicit
   budget drop on the evidence turn. The committed fixtures do not include
   such decisions for the gpt4_a1b77f9c / b46e15ed cases, so the priority
   order (`regimes.py:376-383`) resolves them to assembly-crowding. This
   discrepancy should be reconciled before the paper: either the per-question
   `decisions` payloads in the real run confirm budget-truncation (in which
   case the test fixture needs the same decisions), or the regime label in
   the narrative needs to change to assembly-crowding.

2. `eac54add` is the single regression fixture for the
   evidence-vs-session distinction; see
   `test_eac54add_does_not_misclassify_as_budget_truncation`
   (`tests/test_regime_recalibration.py:344-358`).

### 4.5 In-container mock-mode artifact (reproducible here)

A `--mode mock --full` run of `scripts/run_loop.py` produces
`runs/loop_001/report.json` with the following (reproduced from the file just
written by `python scripts/run_loop.py --mode mock --full`):

- Overall accuracy: **0.4286** (3 of 7 mock instances correct)
- Per-type: `knowledge-update` 1.0, `single-session-user` 1.0,
  `multi-session` 0.333, `temporal-reasoning` 0.0
- Failure histogram: 1 scoring-error, 1 retrieval-signal-gap,
  1 budget-truncation, 1 assembly-crowding, 0 assemble-internal
- Drafted target: `assembly-crowding` (tie-broken by `_choose_target`,
  `behaviors.py:632-645`)
- Drafted transform: `stub_topk_boost` (StubAuthor library entry,
  `hypothesize.py:67-84`)
- All three attempts discarded with reason
  `"target regime did not shrink: target_delta=0"`
- `stopped.reason = max_consecutive_discards`
- `stopped.named_wall = "retrieval-signal-gap=1 → signal change (better
  embedder / scorer); scoring-error=1 → fix the scoring-step exception
  (e.g. input truncation before embedding)"`
- n_events: **74**

This is what's reproducible here. The real-mode loop run with baseline 0.78
the user describes is a different artifact — same code, different fixture.

---

## 5. Engineering hardening (bug ledger)

Every bug below was invisible on the synthetic fixture and surfaced only when
the loop ran against real LME data through the real reader + real judge. The
methodology lesson — tests built from real artifacts catch what synthetic
fixtures don't — is the one belonging in the paper.

### Bug A — embedder BadRequestError on long turns
Commit `5338209` ("fix: truncate embedding inputs to 8000 tokens; disable
ID-based type check in load_split (band-aid)"). Long LME turns blew past
`text-embedding-3-small`'s 8192-token limit, the OpenAI API returned 400, the
score behavior raised, the chain produced an empty context, and the reader
emitted empty answers that the judge marked wrong without anyone seeing the
scoring exception. Fix: `_truncate_for_embedding()` at
`src/regimes/agent/embedders.py:43-61` clamps to 8000 tokens (uses `tiktoken`
when available, falls back to char-based clamp). The score-error is now
surfaced as `Outcome.score_error` (`src/regimes/eval/types.py:62-68`) and
classified as the non-optimizable `scoring-error` regime.

### Bug B — `load_split` type-coverage check on opaque real IDs
Same commit `5338209`. The split loader's per-type / abstention coverage
check derived `question_type` by string-prefix on the qid. Real LME qids are
opaque hashes (`078150f1`, `07b6f563`, …) so the prefix never matched. Fix:
disjointness + duplicate guards retained; the type-coverage check is disabled
with a TODO to bake types into split.json at build time (`split.py:103-105`).

### Bug C — agent registry fragility + judge subprocess relative-path crash
Commit `d9a8033` ("fix two eval-wiring bugs in the loop's real-eval path").

- **C1**: `agent.retrieve()` previously called `clear_registry()` then
  registered its behaviors back into the global registry and built
  `Runtime(graph)` relying on the global to resolve them. Inside the loop's
  runtime body, any concurrent registry mutation (or the loop's own
  registration of its phase behaviors) silently re-cleared the agent's
  registry, producing empty contexts. Fix: pass `Runtime(behaviors=...)`
  explicitly to make the agent's runtime self-contained
  (`src/regimes/agent/agent.py` and the same pattern used at
  `src/regimes/loop/runner.py:90-91`). The empty-context case is also
  now surfaced as an explicit `Outcome.error` rather than letting it look
  like a reader saying "I don't know" (`real.py:486-491`).
- **C2**: `LMEJudge.judge()` runs the upstream `evaluate_qa.py` as a
  subprocess with `cwd=<lme_checkout>`, but callers (the loop's `_RD`
  wrapper, `scripts/run_loop.py:121-126`) handed it paths like
  `runs/loop_001/sub_1/hypotheses.jsonl` relative to the regimes repo.
  Subprocess resolved them against the LME cwd → `FileNotFoundError`. Fix:
  resolve all paths to absolute before the subprocess runs
  (`src/regimes/eval/real.py:161-167`).

### Bug D — every failure collapsed to `assemble-internal`
Commit `d01cc4c` ("fix Bug D: regime detectors collapsed every failure into
assemble-internal"). The original detector was
`bool(o.gold_selected()) and not o.correct` — ANY gold-session turn in
`selected_turn_ids` fired it. The seed phase reliably picks at least one
gold-session turn (often non-evidence filler that scored high), so this
detector matched every failure. Combined with the previous `TOP_K=5` being
too narrow for assembly-crowding (gold ranked at 8–16 didn't count as
"well-ranked"), every actionable regime fell through to the catch-all.

Fix: `WELL_RANKED_K = 20` consolidated window
(`regimes.py:63-69`); coverage-based split for assemble-internal vs
assembly-crowding using `ASSEMBLE_COVERAGE_FLOOR = 0.5`
(`regimes.py:186-256`); priority order reshuffled so assemble-internal is
last among the meaningful regimes, not slot 2. Regression fixtures pinned in
`tests/test_regime_recalibration.py`.

### Bug E — session-level signals masked evidence-level reality
Commit `0575811` ("fix Bug E: detectors used session-level signal; promote
to evidence-turn granularity"). After Bug D the detectors used session-level
helpers parameterized by `answer_session_ids`. Failures swung from
assemble-internal (Bug D) into budget-truncation (Bug E) because the agent
selected high-scoring non-evidence filler from the gold session, and one of
those filler turns showed up in `decisions[reason='budget']`. Detector saw
"gold session turn dropped at budget" → fired budget-truncation. Impossible
in reality — the actual evidence at rank 126 was never seeded.

Fix: evidence-turn helpers on `Outcome`
(`src/regimes/eval/types.py:120-181`); detectors prefer evidence-level when
`gold_evidence_turn_ids` is non-empty
(`regimes.py:91-126, 154-183, 229-235, 273-275`); evidence ids extracted
from both LME variants in `real.py:369-416`.

### Bug F — never-retrieved cases were optimizable in the persisted record
Commit `4d7a940` ("fix Bug F: persist evidence signals; never-retrieved cases
cannot be optimizable"). Persisted per-question records held only
`qid/correct/regime` — none of the detector's input signals — so a label
of "budget-truncation" was unauditable on disk. Worse, the detector itself
was loose: it required only `truncated=True` and a session-level
`_gold_dropped_at_budget`, so any evidence-turn-at-rank-126 case could leak
into the optimizable bucket. Fix: (a) `_outcome_summary()` writes the
evidence-level signals on every `baseline.recorded` payload
(`src/regimes/loop/behaviors.py:46-108`); (b) `detect_budget_truncation`
hard-requires non-empty `evidence_ranked_top_k(WELL_RANKED_K)`
(`regimes.py:209-235`).

### Bug G — LME judge results-file parser format drift
Commits `627cc9d` ("fix LMEJudge results-file parser: parse the array format
upstream actually writes") and `b175fef` ("fix: unwrap nested autoeval_label
dict in results parser"). Upstream `evaluate_qa.py` writes the results as a
pretty-printed JSON ARRAY, not JSON-lines, and the `autoeval_label` field is
currently a nested `{"model": ..., "label": true}` dict rather than a bare
"0"/"1" string. Parser at `src/regimes/eval/real.py:200-285` accepts:

- whole-file JSON array
- whole-file JSON object wrapping a list under `results` / `evaluations` /
  `items`
- JSON-lines fallback
- `autoeval_label` as boolean, int, or "0"/"1" / "correct"/"wrong" string,
  including the nested-dict case (`real.py:275-277`)

It also reconstructs `question_id` positionally from `hypotheses.jsonl` if
upstream didn't echo it (`real.py:255-273`), since current LME records do
not reliably carry the qid.

### Bug H (recurring lesson)
Every bug above was invisible on `fixtures/synthetic_lme.json`. A & C surfaced
on real corpora (long turns / cross-runtime registry). D & E & F surfaced on
real evidence distributions (gold-session contains non-evidence filler). G
surfaced on the real upstream judge's output format. The synthetic fixture
exercised every code path; it did not exercise the data distribution. The
methodology lesson the paper should record: **build the test fixture from
real artifacts as soon as one real run exists**, even if the fixture is just
a hand-extracted subset.

---

## 6. The loop's first full run + open questions

The loop's first full `--full` run (no `pause_after="histogram"`) completed
autonomously. The narrative (user-reported, from a real-mode run not in this
repo):

1. Drafted 3 score-transform candidates targeting `budget-truncation`
   (the dominant optimizable+seam-reachable regime, count 4).
2. Discarded all 3. Discard reasons recorded in `transform_log`:
   either `target regime did not shrink` (`target_delta >= 0`) or `target
   shrank by 1 but overall regressed −0.02`. The promotion rule is at
   `src/regimes/loop/gates.py:307-329` — eligibility requires
   `target_delta < 0`, no multi-session regression, and no overall regression.
3. Stopped on `max_consecutive_discards` (default 3, `runner.py:59`;
   guard at `behaviors.py:505-518`).
4. Named the wall: `assemble-internal=6 → assemble() internals change
   (reader prompt / context format); retrieval-signal-gap=1 → signal change
   (better embedder / scorer)` — matching the `_name_wall()` template at
   `src/regimes/loop/behaviors.py:605-629`.

### 6.1 The CAVEAT — author was StubAuthor, not a real LLM

**This run validates the loop mechanism, not the action space.** Per the
narrative, the `transform_log` shows the author was `StubAuthor` and the
drafted transform was `stub_demote_low` — a hand-written halve-below-median
function whose source lives at `hypothesize.py:88-99`. No LLM-authored code
was ever drafted, gated, or evaluated.

What the run did establish:

- The phase chain drains end-to-end without manual intervention.
- The static, sandbox, and eval-diff gates execute and produce
  decisions.
- The deterministic promotion rule rejects when the target regime doesn't
  shrink AND/OR overall regresses.
- `max_consecutive_discards` fires and terminates the loop.
- `loop.stopped` payload names the wall in the documented shape.
- Held-out discipline holds: CONFIRM was never touched because no
  transform passed promotion.

What it did **not** establish:

- Whether the score-transform seam has "juice" — i.e. whether any
  well-authored function over the existing score dict can shrink
  budget-truncation without regressing multi-session. The action space's
  empirical reach is still open.

### 6.2 Immediate next step

Run the loop with `LLMAuthor` instead of `StubAuthor` on the same OPTIMIZE-50.
`LLMAuthor.draft()` (`hypothesize.py:178-223`) is wired and validated at
construction; the change is one line in `run_loop.py`. Treat `transform_log`
as the deliverable — the best-of-N audit IS the result, headline number from
CONFIRM-100 attached to the single transform (if any) that passes.

### 6.3 The deferred lever — "option-2" preprocessing/assembly seam

The named wall lists 6 `assemble-internal` + 1 `retrieval-signal-gap` —
**7 of 11 failures the score-transform seam structurally cannot touch**.
Score-transforms re-weight existing scores; they cannot inject a signal that
isn't there, change what the reader sees, or change what `assemble()` does
with the budget. A second action space — a preprocessing / assembly seam —
would be needed to reach those regimes: reference-date injection into the
question, a better embedder (replace `text-embedding-3-small`), or a
context-format change inside `assemble()`. This is outside the current scope;
the loop's "named wall" output is intentionally the boundary of the current
scope, not a problem statement for it.

---

## 7. Reproducibility

### 7.1 What is pinned

| Knob | Value | Where |
|---|---|---|
| `activegraph` version | `1.0.5.post2` | `pyproject.toml:11` |
| Python | `==3.11.*` | `pyproject.toml:8` |
| Reader | `claude-sonnet-4-5`, T=0 | `src/regimes/eval/real.py:55-99` |
| Judge | `gpt-4o`, upstream `evaluate_qa.py` | `src/regimes/eval/real.py:120-197` |
| Embedder | `text-embedding-3-small` | `src/regimes/agent/embedders.py:127` |
| Split seed | 42 | `config/split.json:5` |
| Split source SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` | `config/split.json:3` |
| Loop frozen clock | `2026-01-01T00:00:00Z` | `src/regimes/loop/runner.py:58` |
| Loop run_id | `regimes-loop-{iteration_id}` | `src/regimes/loop/runner.py:88` |
| Agent FrozenClock + IDGen | per ingest, fresh | `src/regimes/agent/agent.py:98-108` |
| Aggregate version tag | `regimes-eval-real-v1` | `src/regimes/eval/real.py:45, 589` |

### 7.2 What determinism the tests actually cover

Two property tests live in `tests/test_agent_determinism.py`:

- `test_reingest_equality_first_instance` and
  `test_reingest_equality_three_distinct_instances` assert byte-identical
  event logs across two re-ingests of the same instance under
  `FrozenClock + IDGen()` (`tests/test_agent_determinism.py:52-73`).
- A third test (line ~113) asserts byte-identical full-run logs across runs.

These cover the ingest + retrieval chain. The reader is NOT covered — see
below.

### 7.3 The one stochastic element

The reader (`claude-sonnet-4-5`, T=0) is the only stochastic element in the
real-mode chain. Temperature 0 reduces but does not eliminate API-side
variance. The author observed roughly ±2 points of overall accuracy across
real-mode runs on the same OPTIMIZE split (`0.76` ↔ `0.78`). All other
phases — agent, gates, attribution, stop — produce byte-equal event payloads
on re-run modulo this reader noise.

The mock-mode loop (`MockEval` + `StubAuthor` + `FakeReader/FakeJudge`)
is fully deterministic; the in-container reproduction (§4.5) produces the
same numbers every time.

### 7.4 What is NOT pinned (and should be before the paper)

- No model snapshot pin. `claude-sonnet-4-5`, `gpt-4o`, and
  `text-embedding-3-small` are model families that can update under the same
  string. Recording the API-returned snapshot ID in the run aggregate would
  close this.
- No upstream LME submodule SHA recorded in the run aggregate. The judge
  shells out to `evaluate_qa.py` from `third_party/longmemeval`; the
  submodule commit SHA is implicit, not logged.
- No `repo SHA` field in the run aggregate. `LoopReport.iteration_id` is the
  only run-identifier; `subprocess.run(['git', 'rev-parse', 'HEAD'])` at
  baseline-recorded time would suffice.

These are aspirational items, not implemented. The README and bug-ledger
commits mention "manifest pinning" as a property; the only manifest
currently present is `split.json`, which pins source path + sha256 +
seed but not the rest.

---

## 8. Test posture

`python -m pytest` in this environment reports **157 passed, 3 failed**
(2025-05-23). The 3 failures are all in `tests/test_split.py` and all
related to the same root cause: `config/split.json` was repointed at the
real LME source SHA (commit `5338209`) while three of the test cases still
expected the synthetic-fixture SHA. They are:

- `test_committed_split_required_types_in_both`
- `test_loader_rejects_missing_abstention`
- `test_split_generator_is_deterministic`

Commit `66dc55e` ("restore eval tests after split.json was repointed at real
LME data") restored most of the affected tests but did not finish these three.
Resolving them is a small follow-up — either regenerate
`fixtures/synthetic_lme.json` to match the committed split.json's source
sha256, or rebuild split.json against the synthetic fixture in a separate
test-only file.

Everything else (130+ tests across agent determinism, embedder, eval wiring,
LME parser, loop gates, runtime, mock eval, regime detectors, regime
recalibration regression fixtures, signal-gap priority) is green.

---

## 9. Open items for the paper

1. **Reconcile §4.4 regime labels.** Pin the per-question `decisions`
   payloads from the real run so the gpt4_a1b77f9c / b46e15ed cases
   either confirm budget-truncation in the fixture or the narrative
   moves to assembly-crowding.
2. **Run LLMAuthor on OPTIMIZE-50.** This is the experiment the loop was
   built for; nothing in §6 measures the score-transform action space's
   reach. Until this runs, the "the loop found nothing" result is about
   StubAuthor's library, not about the seam.
3. **Pin model snapshots, LME submodule SHA, and repo HEAD into the run
   aggregate.** §7.4.
4. **Move `runs/` into a release artifact path.** It is gitignored today,
   which means every external reader (including this document) is dependent
   on the author transcribing numbers correctly. A `runs/published/` track
   that gets committed (or a release-asset upload) would close this.
5. **Decide whether to implement the option-2 preprocessing/assembly
   seam.** Without it, the score-transform action space cannot reach the
   majority of the named wall. With it, the loop has a second
   action-space to grow into when the first is exhausted.
