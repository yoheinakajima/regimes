# Reader-prompt-transform result — replicated

Two independent real runs, LongMemEval-S. Held-out confirm read against ~0.02 noise bar.

Run 1 (run_2026-05-30): baseline 0.70, OPTIMIZE +0.18, target_delta -7, HELD-OUT confirm +0.04
Run 2 (run_2026-05-30b): baseline 0.78, OPTIMIZE +0.08, target_delta -2, HELD-OUT confirm +0.03

Key finding: OPTIMIZE deltas diverge (0.18 vs 0.08 = baseline luck + overfit); held-out deltas converge (0.04 vs 0.03). Stable generalizing effect ~+0.03-0.04.

Both promoted transforms independently converge on the same reconciliation principles: trust retrieved context (anti-hedging), count/synthesize across all sessions, resolve relative time against session dates. Run 1 also carried question-type-specific instructions (overfit) that inflated OPTIMIZE and did not transfer.

Score-transform seam (budget-truncation) regresses on held-out in both runs. Reader-prompt seam (assemble-internal/reconciliation) generalizes.
