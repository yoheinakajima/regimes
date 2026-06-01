import json, sys
from math import comb

def mcnemar_exact(b, c):
    # exact two-sided binomial test on discordant pairs (b=right->wrong, c=wrong->right)
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    # P(X<=k) + P(X>=n-k) under p=0.5, two-sided
    tail = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
    p = min(1.0, 2*tail)
    return p

def analyze(path, label):
    r = json.load(open(path))
    for p in r.get('promotions', []):
        if 'reader' not in p['name']: continue
        b = {o['question_id']: o['correct'] for o in p.get('confirm_baseline_outcomes', [])}
        t = {o['question_id']: o['correct'] for o in p.get('confirm_transform_outcomes', [])}
        common = set(b) & set(t)
        if not common: 
            print(f"{label}: no per-q confirm data"); continue
        w2r = sum(1 for q in common if not b[q] and t[q])
        r2w = sum(1 for q in common if b[q] and not t[q])
        n = len(common)
        base_acc = sum(b[q] for q in common)/n
        trans_acc = sum(t[q] for q in common)/n
        pval = mcnemar_exact(r2w, w2r)
        print(f"\n{label} (n={n}, held-out)")
        print(f"  baseline {base_acc:.3f} -> transform {trans_acc:.3f}  (delta {trans_acc-base_acc:+.3f})")
        print(f"  discordant: wrong->right={w2r}, right->wrong={r2w}")
        print(f"  McNemar exact two-sided p = {pval:.4f}")
        return common, b, t, w2r, r2w
    return None

runs = sys.argv[1:]
labels = ['fixed-1','fixed-2','seed7']
pooled_w2r = pooled_r2w = pooled_n = 0
for path, lab in zip(runs, labels):
    res = analyze(path, lab)
    if res:
        _,_,_,w,rw = res
        pooled_w2r += w; pooled_r2w += rw

# pooled (note: fixed-1 and fixed-2 share a split, so pooling is approximate)
print(f"\n=== POOLED discordant (note: fixed runs share a split) ===")
print(f"  wrong->right={pooled_w2r}, right->wrong={pooled_r2w}")
print(f"  McNemar exact two-sided p = {mcnemar_exact(pooled_r2w, pooled_w2r):.4f}")
