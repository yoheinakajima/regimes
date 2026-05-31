import json, sys
from collections import defaultdict

def tables(path):
    r = json.load(open(path))
    print(f"\n=== {path} ===")
    for p in r.get('promotions', []):
        b = {o['question_id']: o for o in p.get('confirm_baseline_outcomes', [])}
        t = {o['question_id']: o for o in p.get('confirm_transform_outcomes', [])}
        common = set(b) & set(t)
        if not common:
            print(f"  {p['name']}: no per-question CONFIRM outcomes"); continue
        print(f"\nTransform: {p['name']}  confirm_delta={p.get('confirm_delta')}  n={len(common)}")
        print("\n[1] PER-TYPE held-out delta")
        print(f"  {'type':<28}{'base':>6}{'trans':>7}{'delta':>7}")
        for ty in sorted({b[q]['question_type'] for q in common}):
            qs=[q for q in common if b[q]['question_type']==ty]
            bc=sum(b[q]['correct'] for q in qs); tc=sum(t[q]['correct'] for q in qs)
            print(f"  {ty:<28}{bc:>6}{tc:>7}{tc-bc:>+7d}")
        ab=[q for q in common if b[q].get('is_abstention') or t[q].get('is_abstention')]
        if ab:
            bc=sum(b[q]['correct'] for q in ab); tc=sum(t[q]['correct'] for q in ab)
            print(f"\n[2] ABSTENTION  n={len(ab)}  base={bc}  trans={tc}  delta={tc-bc:+d}")
        else:
            print("\n[2] ABSTENTION  none flagged")
        w2r=[q for q in common if not b[q]['correct'] and t[q]['correct']]
        r2w=[q for q in common if b[q]['correct'] and not t[q]['correct']]
        print(f"\n[3] FLIPS  wrong->right {len(w2r)} | right->wrong {len(r2w)} | net {len(w2r)-len(r2w):+d}")
        print("\n[4] LOCALIZATION (net flips by baseline regime)")
        loc=defaultdict(lambda:[0,0])
        for q in w2r: loc[b[q].get('regime','?')][0]+=1
        for q in r2w: loc[b[q].get('regime','?')][1]+=1
        for reg,(u,d) in sorted(loc.items()):
            print(f"  {reg:<28} +{u} / -{d}  (net {u-d:+d})")

for p in sys.argv[1:]: tables(p)
