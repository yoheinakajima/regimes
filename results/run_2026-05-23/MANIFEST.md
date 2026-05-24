# Run Manifest — 2026-05-23

| Field | Value |
| --- | --- |
| Date | 2026-05-23 |
| Command | `python scripts/run_loop.py --mode real --full --lme-data ../activegraph-longmemeval/data/longmemeval_s_cleaned.json` |
| Author model | `claude-sonnet-4-6` (via `BEHAVIORDRAFTS_MODEL`) |
| Reader / eval model | `gpt-4o` (per `sub_*/eval.log`) |
| Code SHA (approx) | `32fcd47` — *approximate; `runs/` is gitignored and not version-locked to commits, so the executing tree may include uncommitted edits relative to this SHA* |
| Artifact snapshot commit | `a5a5a64` (`save run_2026-05-23 artifacts`) |
| Iteration id | `loop-001` |
| n_events | 149 |
| Wall clock | ~80 min (API latency dominated; irrelevant to results) |

## Files

- `report.json` — full loop record (histogram, baseline outcomes, transform_log, promotions, discards, attributions, stopped).
- `sub_1/` … `sub_10/` — per-eval `aggregate.json` + `eval.log`. **Only `sub_1`–`sub_6` correspond to this run; `sub_7`–`sub_10` are leftover artifacts from earlier runs that reused the same `loop_001` directory.** Authoritative candidate list is `report.json::transform_log`.

## Provenance caveat

The `runs/` directory the loop wrote into is gitignored; these copies under `results/run_2026-05-23/` are the source of truth, but they were captured after the fact, so exact code state at execution time is inferred from commit history rather than a tag. Treat `32fcd47` as the nearest commit, not a verified pin.
