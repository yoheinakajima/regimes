# Preliminary Report — Loop run 2026-05-23 (`loop-001`)

> **PROVISIONAL.** This run executed on a corrupted venv (see Caveats §a). All numbers below are sourced directly from `report.json` and `sub_*/aggregate.json` in this directory, but should be re-confirmed against a clean-venv rerun before drawing firm conclusions.

## What was tested

First end-to-end run of the autonomous loop with:

- a **real LLM author** drafting score-transforms (`claude-sonnet-4-6` via `BEHAVIORDRAFTS_MODEL`), and
- **CONFIRM-100 held-out validation** wired into `run_loop.py` (commit `1af577b` landed via PR #5 / merge `32fcd47`, the closest commit to execution).

Target action space: score-transforms aimed at `budget-truncation` failures. Reader/eval: `gpt-4o`. Data: 50-question OPTIMIZE split + 100-question CONFIRM split from `longmemeval_s_cleaned.json`.

## Baseline (OPTIMIZE-50)

From `report.json::baseline` and `sub_1/aggregate.json`:

- **Overall accuracy: 0.76** (38 / 50 correct, `n_errors=0`, `n_truncated=50`)
- **Per-type accuracy:**

  | type | acc |
  | --- | --- |
  | knowledge-update | 1.000 |
  | multi-session | 0.846 (11/13) |
  | single-session-assistant | 1.000 |
  | single-session-preference | 0.667 |
  | single-session-user | 1.000 |
  | temporal-reasoning | 0.357 (5/14) |

- **Regime histogram (12 failures / 50):**

  | regime | count | optimizable | seam_reachable |
  | --- | --- | --- | --- |
  | scoring-error | 0 | no | no |
  | retrieval-signal-gap | 1 | no | no |
  | budget-truncation | 4 | yes | yes |
  | assembly-crowding | 0 | yes | yes |
  | assemble-internal | 7 | no | no |
  | unclassified | 0 | no | no |

  Failure qids by regime:
  - **budget-truncation (4):** `b46e15ed`, `gpt4_7abb270c`, `gpt4_a1b77f9c`, `gpt4_f2262a51`
  - **retrieval-signal-gap (1):** `eac54add`
  - **assemble-internal (7):** `8077ef71`, `a3045048`, `fca70973`, `gpt4_2f8be40d`, `gpt4_5438fa52`, `gpt4_8279ba03`, `gpt4_b0863698`

- **Reader non-determinism context.** Across three baseline runs today (same data, same code path), overall came out **0.74 / 0.76 / 0.78**. The 0.04 promotion gate sits inside this noise band; treat single-digit accuracy moves with that in mind.

## Transform attempts (core result)

All six candidates came from a single author session targeting `budget-truncation`. Status, deltas, and OPTIMIZE/CONFIRM outcomes are taken verbatim from `report.json::transform_log`, `::discards`, and `::promotions`.

| # | status | overall_delta | target_delta | confirm_delta | notes (from `reasons` / report) |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | discarded | -0.06 | +3 | — | target grew; multi-session -0.1538; overall -0.06 |
| 2 | discarded | -0.06 | +2 | — | target grew; multi-session -0.0769; overall -0.06 |
| 3 | **promoted** | **+0.04** | **-2** | **-0.01** | only candidate to shrink target; CONFIRM did **not** confirm |
| 4 | discarded | +0.04 |  0 | — | target unchanged (gate: target must shrink) |
| 5 | discarded | +0.04 | +1 | — | target grew |
| 6 | discarded | -0.06 | +3 | — | target grew; multi-session -0.0769; overall -0.06 |

**Pattern.** Several candidates moved overall accuracy on OPTIMIZE (#3/#4/#5 each +0.04 overall) but the *target regime did not shrink* (`target_delta ≥ 0`) — i.e. they fixed other things while leaving budget-truncation untouched. The two that *did* attempt aggressive reweighting (#1, #2, #6) regressed multi-session by -0.08 to -0.15 and overall by -0.06: classic rob-Peter-to-pay-Paul.

Only **#3** cleared every OPTIMIZE gate: overall +0.04, target shrank by 2 (4 → 2 budget-truncation failures), no listed regression. It was promoted. Its **CONFIRM-100 delta was -0.01** — i.e. on the held-out 100-question split, the transform was *slightly worse* than baseline. Supporting eval detail: `sub_5/aggregate.json` reports overall 0.79 and `sub_6/aggregate.json` reports overall 0.80 on n=100, consistent with a -0.01 confirm delta.

Per `report.json::attributions[0]`, the promoted transform recovered 1 question (`gpt4_f2262a51`: budget-truncation → correct) and introduced 1 new failure (`36b9f61e`: correct → assemble-internal; `eac54add`: retrieval-signal-gap → assemble-internal — two transitions, one new wrong answer net).

### Promoted transform — full source (transform_log[2].source)

```python
import math

def transform(scores: dict, graph, question: str, question_date: str) -> dict:
    if not scores:
        return scores

    all_scores = list(scores.values())
    max_score = max(all_scores)
    min_score = min(all_scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    # Extract conversation prefix -> list of (turn_id, score, turn_index)
    prefix_scores = {}
    for turn_id, score in scores.items():
        parts = turn_id.rsplit('#', 1)
        if len(parts) == 2:
            prefix = parts[0]
            try:
                idx = int(parts[1])
            except ValueError:
                idx = 0
            if prefix not in prefix_scores:
                prefix_scores[prefix] = []
            prefix_scores[prefix].append((turn_id, score, idx))

    # For each prefix, find max score in that conversation
    prefix_max = {}
    for prefix, items in prefix_scores.items():
        prefix_max[prefix] = max(s for _, s, _ in items)

    # Compute mean and std for normalization reference
    mean_score = sum(all_scores) / len(all_scores)
    variance = sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)
    std_score = math.sqrt(variance) if variance > 0 else 1.0

    new_scores = {}
    for turn_id, score in scores.items():
        parts = turn_id.rsplit('#', 1)
        boost = 0.0

        if len(parts) == 2:
            prefix = parts[0]
            try:
                idx = int(parts[1])
            except ValueError:
                idx = 0

            conv_max = prefix_max.get(prefix, score)

            # Boost #0 turns (first turns) - they are frequently evidence
            if idx == 0:
                # Stronger boost for #0 turns, scaled by how high the conversation scores
                boost += 0.08 * (1.0 + conv_max)

            # Boost turns from conversations that have at least one high-scoring turn
            # This helps pull up lower-ranked turns from relevant conversations
            if conv_max > mean_score + 0.5 * std_score:
                # Conversation is relevant, boost all its turns slightly
                boost += 0.03 * conv_max

            # Compress scores toward mean slightly to reduce budget wall effect
            # Pull low scores up more than pulling high scores down
            if score < mean_score:
                compression_boost = 0.15 * (mean_score - score)
                boost += compression_boost

        new_scores[turn_id] = score + boost

    return new_scores
```

## Key finding

The OPTIMIZE/CONFIRM split did its job: a transform that cleared every OPTIMIZE gate (overall +0.04, target -2, no listed regression) failed held-out confirmation (CONFIRM-100 Δ = -0.01). That is direct, first-run evidence that score-transform candidates can overfit the OPTIMIZE-50 slice.

More broadly, the score-transform action space did not produce a generalizing improvement against this benchmark in this run:

- The two remaining unreachable regimes named by `report.json::stopped.remaining_regimes` — `budget-truncation` (after the recovery) and `assemble-internal` — are not co-addressable by score reweighting: `assemble-internal` failures are reasoning-bound (evidence is in context, reader still wrong), and `retrieval-signal-gap` is signal-bound (evidence isn't well-ranked at all).
- The loop stopped on `max_consecutive_discards` and emitted a **named wall**: `assemble-internal=9 → assemble() internals change (reader prompt / context format)` (from `report.json::stopped.named_wall`). That is the next seam the loop is pointing at.

## Caveats — do not omit

**(a) Corrupted venv at run time.** `agent.score_embedding` raised `ValueError` twice during execution, traceable to a downgraded `numpy 1.23.3` / `tiktoken 0.3.3` that landed via an accidental cross-repo `pip install`. The baseline still came out clean (overall 0.76, normal histogram, `n_errors=0` in every `sub_*/aggregate.json`), suggesting the errors were isolated rather than systemic — but treat **all numbers in this report as PROVISIONAL pending a clean-venv rerun**.

**(b) `confirm_delta` is recorded but does NOT gate promotion.** The promoted transform's confirm delta was -0.01 and it was promoted anyway. Held-out signal is currently advisory; the gate logic only looks at OPTIMIZE-side overall/target/per-type deltas.

**(c) ~80 minute wall-clock.** Dominated by API latency. Has no bearing on the results above.

## Next steps

1. **Clean-venv confirming rerun.** Repair `numpy` / `tiktoken` / `httpx` to known-good versions, re-execute the same command, and diff `report.json` against this one. If overall/target deltas reproduce, the OPTIMIZE-vs-CONFIRM split finding stands.
2. **Add confirm-gating.** Make negative `confirm_delta` auto-demote (or at minimum require non-negative confirm to promote). This run promoted a candidate that the held-out signal explicitly flagged.
3. **Option-2 seam.** Move past the score-transform action space at the named wall (`assemble-internal=9 → assemble() internals change (reader prompt / context format)`) — i.e. preprocessing / assembly-layer changes, not score reweighting.
