# regimes

An autonomous eval-improvement loop that runs natively on the published
[ActiveGraph](https://docs.activegraph.ai) runtime. The loop diagnoses a failing
agent into a failure-regime taxonomy, routes the dominant regime to the pipeline
seam that can address it, has an LLM author a repair at that seam, and promotes
the repair only if it survives static checks, sandbox execution, in-sample
evaluation, and held-out validation. The loop's entire history — diagnosis,
authored transform, gate verdicts, promotion or discard, stop decision — is an
ActiveGraph event log, not a side database. That is the point of the project:
**an event-sourced runtime makes a controlled improvement loop auditable and
re-homable, and this repo is the demonstration.**

Companion paper: *Regimes: Autonomous Improvement Loops on an Event-Sourced
Agent Runtime, Demonstrated on LongMemEval* (arXiv: TODO — link on posting).
Substrate paper: *The Log is the Agent* ([arXiv:2605.21997](https://arxiv.org/abs/2605.21997)).

## What it does, in one sentence

It diagnoses LongMemEval failures into regimes, routes the dominant
seam-reachable regime to its action seam (score-transform, assembly-transform,
or **reader-prompt transform**), has an LLM author a repair there, validates it
through compile → sandbox → in-sample eval-diff → held-out promotion gates,
keeps it only if it improves a held-out slice without regressing, and stops when
the remaining failures are in a regime the action space cannot reach.

## Headline result

On LongMemEval-S, the dominant failure is not retrieval but **reconciliation**:
the right evidence is already in the assembled context, yet the reader answers
incorrectly (the `assemble-internal` regime). The reader-prompt seam is the only
one that can touch that wall. Across five seeded held-out splits, the loop
discovers reader-prompt repairs that improve final held-out (CONFIRM) accuracy:

| split | held-out delta | w→r / r→w | CONFIRM base→post | McNemar p |
|-------|---------------|-----------|-------------------|-----------|
| seed 5 | +0.10 | 11 / 1 | 0.78 → 0.88 | 0.006 \* |
| seed 7 | +0.08 | 10 / 2 | 0.74 → 0.82 | 0.039 \* |
| seed 11 | +0.06 | 8 / 2 | 0.77 → 0.83 | 0.109 |
| seed 23 | +0.05 | 7 / 2 | 0.71 → 0.76 | 0.180 |
| seed 101 | +0.01 | 7 / 6 | 0.78 → 0.79 | 1.000 |

\* individually significant at α = 0.05. Four of five splits are clearly
positive (+0.05 to +0.10); seed 101 is a near-zero **over-promotion** finding
(the loop kept promoting past the plateau), which is itself a diagnostic result
about the stopping rule, not a failure to hide. The reader is held fixed
(`claude-sonnet-4-6`); the author is the same model, with a cross-author check
(`claude-haiku-4-5` authoring for the sonnet reader) reproducing the gain on the
three strongest splits. The effect is modest by design: the baseline is high
(CONFIRM 0.71–0.88), so a few held-out points is meaningful movement and the
failure analysis is the contribution, not a new benchmark record.

## What this repo does and does not show

**Shows.** That an event-sourced runtime makes the diagnose→route→repair loop
auditable (every step is an event), re-homable across targets (a second
text-to-SQL target runs through the unchanged loop with byte-identical event
logs), and capable of a held-out-validated, modest improvement on one benchmark
under one reader.

**Does not show.** That event-sourcing is *necessary* (no non-event-sourced
baseline is compared — the demonstrated wins are auditability and clean
re-homing, by construction); cross-task empirical improvement (the SQL target
was not run to a held-out gain); reader-independent transfer (one reader, one
benchmark); that the regime-to-seam *routing* adds value over handing an author
the failed examples with no labels (the primary open question — no no-routing
ablation here); or that guarded operators outperform prose (proposed as future
work, not built). The pooled cross-split count is descriptive only, since all
splits come from the same 500-question pool.

## Three repos, only one runs on the runtime

| Repo | Role |
|------|------|
| **regimes** (this repo) | The loop. Runs **natively on the ActiveGraph runtime**. Every step is an event. |
| **activegraph-longmemeval** ([github](https://github.com/yoheinakajima/activegraph-longmemeval)) | External system-under-test and scoring function. **Never modified.** The loop shells out to it. |
| **activegraph-behaviordrafts** ([github](https://github.com/yoheinakajima/activegraph-behaviordrafts)) | The paper that specifies the lifecycle principle (`draft → static → sandbox → semantic-diff → promotion → disable`). **Not a dependency.** The gates here are reimplemented natively. |

## The action space

The loop edits one of three typed seams, selected by the diagnosed regime:

- **reader-prompt transform** — edits the reader's prompt fragment (the seam
  that reaches `assemble-internal`, the reconciliation wall; this is where the
  headline result comes from).
- **score-transform** — re-weights retrieved-turn scores at the documented seam
  in `src/regimes/agent/transforms.py`, between scoring and assembly (targets
  `budget-truncation`; generalized in only 1 of 13 attempts — see the paper's
  §5.8).
- **assembly-transform** — reorders or filters the selected turns.

All three are executable patch objects gated by static analysis: a reader-prompt
transform may edit values but not fabricate keys and may inject at most 2000
characters; a score-transform may re-weight, blend, penalize, or filter scores
but **cannot** edit `assemble()` internals, touch the filesystem or network,
mutate the graph, or reach the scorer or dataset. The static gate is the entire
safety model.

## Held-out discipline (READ THIS)

The loop refuses to start without a frozen OPTIMIZE/CONFIRM partition.

- **OPTIMIZE** (50 question_ids, stratified by `(question_type, is_abstention)`)
  is the only set used for diagnosis, sandbox, and in-sample eval-diff.
- **CONFIRM** (100 question_ids, same stratification, disjoint from OPTIMIZE) is
  touched **once per promotion** and only to gate/report.

At startup the loop asserts `OPTIMIZE ∩ CONFIRM = ∅` and raises
`activegraph.ConfigurationError` if it doesn't hold (a caller-fixable
construction error, not a logged event). Reporting an OPTIMIZE-tuned number as
the result would be overfitting; the headline is the CONFIRM number after a
promotion.

The promotion gate computes `confirm_delta` as a **marginal over the current
deployed state**: it installs the candidate, measures CONFIRM, reverts only that
candidate (leaving prior promotes installed), and measures again. Per-promote
deltas therefore do not telescope to the final state — each is a fresh
measurement against a deployed state that, because later transforms re-work the
same questions, does not compound. See the paper's §3.5 and §5.7.

## Reproducing the paper

```
# 1) Real LME data + the upstream judge submodule
cd ../activegraph-longmemeval
make setup && make data
git submodule update --init --recursive

# 2) Build a seeded OPTIMIZE/CONFIRM split against real LME data
cd /path/to/regimes
python scripts/build_split.py --split-seed 7 --source ../activegraph-longmemeval/data/longmemeval_s_cleaned.json

# 3) Keys
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

# 4) Run the loop end to end on one split
python scripts/run_loop.py --mode real --full --split-seed 7 --lme-data ../activegraph-longmemeval/data/longmemeval_s_cleaned.json
```

Each of the five reported splits is reproduced by changing `--split-seed`
to one of `{5, 7, 11, 23, 101}`.

### Result → artifact map

| Paper element | Committed artifact |
|---------------|--------------------|
| Five-split results (Table 1b) | `results/run_seed{5,11,23,101}/`, `results/run_2026-05-31_seed7/` |
| Per-question seed-7 outcomes (the §5.6 tables) | seed-7 report (baseline + transform, with `is_abstention` and regime labels) |
| Promoted reader-prompt transforms (Appendix A) | `results/run_seed*/promoted_reader_prompt_transform.py` |
| Significance (McNemar exact) | `scripts/significance.py` (stdlib `math.comb`; scipy is not a dependency) |
| Held-out tables | `scripts/extract_confirm_tables.py` |
| Multi-seed summary | `results/MULTISEED_FINDINGS.md` |
| Split definitions | `config/split.seed{5,7,11,23,101}.json` |
| Regime classifier | `src/regimes/loop/regimes.py` |
| Held-out promotion gate | `src/regimes/loop/behaviors.py` |
| Author (reader-prompt + score transforms) | `src/regimes/loop/hypothesize.py` |
| Reader / eval bridge | `src/regimes/eval/real.py` |

### What is and isn't committed

Committed: per-question outcome files, analysis scripts, split definitions,
promoted transform sources, and `MULTISEED_FINDINGS.md` — enough to re-derive
every reported statistic. **Not** committed: the content-addressed response
cache (available on request) and the LongMemEval-S data itself (external,
pulled from HuggingFace). So you can recompute every number from the committed
outcomes, but regenerating those outcomes by replay needs the cache, and fresh
model calls are not promised to match (hosted-model nondeterminism; see the
paper's §6). The two earliest fixed-split runs saved aggregate deltas only —
their per-question outcomes were not retained, which is why the detailed tables
are computed on the fresh splits.

Pinned versions: `activegraph 1.0.5.post2` and `click >=8.1` are the only
runtime dependencies; the `eval` extra pulls `anthropic >=0.34` and
`openai >=1.40` (committed runs used `openai 2.38.0`). `numpy` is not a
dependency and `tiktoken` is an optional lazy import (token clamping in the
OpenAI embedder, with a fallback when absent). Python 3.11.

## The runtime-native agent

`regimes.agent` is a retrieval system implemented as four behaviors on the real
`activegraph` runtime — this is what the loop optimizes:

```
question.asked
  -> @behavior(on=["question.asked"])    agent.score_lexical      -> turns.scored
  -> @behavior(on=["turns.scored"])      agent.transform_pipeline -> turns.transformed
  -> @behavior(on=["turns.transformed"]) agent.expand_temporal    -> turns.expanded
  -> @behavior(on=["turns.expanded"])    agent.assemble           -> context.assembled
```

Real package APIs throughout: ingest emits via `graph.add_object(type="turn", ...)`
and `graph.add_relation(...)`; behaviors are decorated with
`@activegraph.behavior(...)` and registered through `_REGISTRY`; the runtime is
`activegraph.Runtime(graph)`, driven by `runtime.run_until_idle()`. Determinism
is wired by `FrozenClock`, a fresh `IDGen()` per ingest, and a stable `run_id`;
re-ingest equality and full-log byte-equality across runs are property-tested.

## Regime taxonomy (deterministic detectors)

| Regime | Optimizable? | Seam-reachable? | Detector |
|--------|--------------|-----------------|----------|
| `scoring-error` | no | no | scoring step raised, or gold turns absent from the scores dict |
| `assemble-internal` | no\* | yes (reader-prompt) | gold selected but answer still wrong — the reconciliation wall |
| `budget-truncation` | yes | yes | gold turn in `decisions` with `included=False, reason='budget'` |
| `assembly-crowding` | yes | yes | gold ranked in top-k but not selected |
| `retrieval-signal-gap` | no | no | gold ranked outside the top-20 — signal misses it |
| `unclassified` | no | no | catch-all; LLM-proposed regimes append after this one |

\* `assemble-internal` is reachable by the reader-prompt seam even though it is
not a retrieval-side fix; the classifier is an **offline oracle** over gold
evidence locations (a diagnostic instrument, not a deployment-time signal — see
the paper's §6).

The classifier reads gold evidence locations, never gold answers; correctness is
scored separately by LongMemEval's unmodified judge (`gpt-4o-2024-08-06`), so
diagnosis and correctness are independent.

## Failure model

Follows ActiveGraph's. Caller-fixable construction errors (bad config,
overlapping split, missing LME checkout when real-eval is requested) **raise**.
Everything during a loop run (a transform crashing in sandbox, an eval failing, a
gate rejecting) is a **logged event** with the error in the payload, and the loop
continues.

## Setup

```
pip install -e .[dev]
pytest -q
```

## Repo layout

```
src/regimes/
  split.py                   # load_split() + invariant guard
  agent/                     # runtime-native four-behavior retrieval agent
  eval/                      # RealEval wrapper (LME harness bridge)
  loop/
    events.py                # loop event vocabulary
    regimes.py               # taxonomy + deterministic detectors + histogram
    hypothesize.py           # StubAuthor + LLMAuthor (Claude); reader-prompt + score transforms
    gates.py                 # static / sandbox / eval-diff / promotion
    attribute.py             # structural diff over two EvalResults
    behaviors.py             # one @behavior per loop phase; held-out promotion gate
    runner.py                # run_loop(): seed -> drain -> report
config/                      # frozen OPTIMIZE+CONFIRM splits; committed
results/                     # committed per-question outcomes + analysis outputs
scripts/                     # build_split, run_loop, significance, extract_confirm_tables
tests/                       # pytest
```

## License

Apache-2.0.
