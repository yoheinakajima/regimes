#!/usr/bin/env python3
"""Exact McNemar significance for the Regimes held-out CONFIRM results.

Recomputes per-split discordant pairs (b = wrong->right, c = right->wrong)
from each report's committed per-question CONFIRM outcomes, using the same
`['correct']` test as confirm_tables.py, against the split's FINAL deployed
state (the last promotion's outcomes), then computes the exact two-sided
McNemar p-value with the binomial formula (stdlib math.comb only).

Per-split p-values are the primary evidence. The pooled value sums
discordances across the five fresh splits drawn from one 500-question pool,
so it is reported as a DESCRIPTIVE same-pool summary, not independent-sample
evidence, matching the paper.

Usage:
  python3 scripts/significance.py results/run_seed5/report.json \\
      results/run_2026-05-31_seed7/report.json \\
      results/run_seed11/report.json results/run_seed23/report.json \\
      results/run_seed101/report.json
"""
import json, sys
from math import comb


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar p from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * P(X <= k) under Binomial(n, 0.5), capped at 1.0
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def final_state_flips(report_path):
    """b, c from the split's final deployed state: baseline vs the LAST
    promotion's transform outcomes (the cumulative deployed stack)."""
    r = json.load(open(report_path))
    promos = [p for p in r.get('promotions', [])
              if p.get('confirm_baseline_outcomes') and p.get('confirm_transform_outcomes')]
    if not promos:
        return None  # aggregate-only run (e.g. the earlier fixed-split runs)
    last = promos[-1]
    base = {o['question_id']: o for o in last['confirm_baseline_outcomes']}
    tran = {o['question_id']: o for o in last['confirm_transform_outcomes']}
    common = set(base) & set(tran)
    b = sum(1 for q in common if not base[q]['correct'] and tran[q]['correct'])
    c = sum(1 for q in common if base[q]['correct'] and not tran[q]['correct'])
    return b, c, len(common)


def main(paths):
    print(f"{'split':<34}{'b(w->r)':>9}{'c(r->w)':>9}{'n':>6}{'McNemar p':>14}")
    print("-" * 72)
    pool_b = pool_c = 0
    pool_b_excl = pool_c_excl = 0
    for path in paths:
        res = final_state_flips(path)
        label = path.split('/')[-2] if '/' in path else path
        if res is None:
            print(f"{label:<34}{'--':>9}{'--':>9}{'--':>6}{'aggregate-only':>14}")
            continue
        b, c, n = res
        p = mcnemar_exact_two_sided(b, c)
        print(f"{label:<34}{b:>9}{c:>9}{n:>6}{p:>14.4g}")
        pool_b += b; pool_c += c
        if 'seed101' not in label:
            pool_b_excl += b; pool_c_excl += c
    print("-" * 72)
    pp = mcnemar_exact_two_sided(pool_b, pool_c)
    pe = mcnemar_exact_two_sided(pool_b_excl, pool_c_excl)
    print(f"{'POOLED (all, DESCRIPTIVE same-pool)':<34}{pool_b:>9}{pool_c:>9}{'':>6}{pp:>14.4g}")
    print(f"{'POOLED (excl seed101, DESCRIPTIVE)':<34}{pool_b_excl:>9}{pool_c_excl:>9}{'':>6}{pe:>14.4g}")
    print("\nNote: pooled values sum discordances across splits from one 500-question")
    print("pool; they overstate effective n and are descriptive, not independent-sample")
    print("evidence. Per-split p-values are the primary claim. Earlier fixed-split runs")
    print("(aggregate-only) carry no per-question outcomes and are omitted from pooling.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
