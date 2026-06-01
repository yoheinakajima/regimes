# Multi-seed replication (5 fresh draws) — final state

| draw | OPT base | CONFIRM base | post | Δ | w→r | r→w | McNemar p | #promotes |
|---|---|---|---|---|---|---|---|---|
| Run 3 (seed 7) | 0.80 | 0.74 | 0.82 | +0.08 | 10 | 2 | 0.039 | 1 |
| seed 11 | 0.74 | 0.77 | 0.83 | +0.06 | 8 | 2 | 0.109 | 2 |
| seed 23 | 0.72 | 0.71 | 0.76 | +0.05 | 7 | 2 | 0.180 | 1 |
| seed 5  | 0.58 | 0.78 | 0.88 | +0.10 | 11 | 1 | 0.006 | 4 |
| seed 101| 0.88 | 0.78 | 0.79 | +0.01 | 7 | 6 | 1.000 | 5 |

Pooled (final state, all 5): w→r=43 r→w=13 → McNemar exact p = 7.3e-05
Sensitivity A (excl seed101 over-promote): w→r=36 r→w=7 → p = 9.0e-06
All draws promoted only assemble-internal reader-prompt transforms.
All budget-truncation score-transforms discarded across all seeds.
seed101 = over-promotion finding (high baseline, loop promoted noise after gains plateaued).
