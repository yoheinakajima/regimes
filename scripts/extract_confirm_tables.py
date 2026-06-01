#!/usr/bin/env python3
"""Extract per-type / abstention / flip / localization tables from saved Regimes reports.
Reads CONFIRM-side per-question outcomes for baseline vs the promoted transform.
No API, no rerun — operates on saved report.json files."""
import json, sys
from collections import defaultdict

def load(path):
    with open(path) as f:
        return json.load(f)

def find_confirm_blocks(report):
    """Locate baseline-confirm and promoted-transform-confirm per-question outcomes.
    Reports vary; we probe likely locations and report what we find."""
    # Dump top-level keys so we can see structure if probing fails
    keys = list(report.keys())
    # Heuristics: look for a promoted transform entry that carries confirm-side outcomes
    promoted = [t for t in report.get('transform_log', [])
                if t.get('status') == 'promoted' and t.get('transform_type') == 'reader-prompt-transform']
    return keys, promoted

def per_question(outcomes):
    """Normalize a list of outcome dicts to {qid: (correct, qtype, regime, is_abstention)}."""
    out = {}
    for o in outcomes:
        qid = o.get('question_id') or o.get('qid') or o.get('id')
        if qid is None:
            continue
        out[qid] = (
            bool(o.get('correct')),
            o.get('question_type') or o.get('qtype') or '?',
            o.get('regime') or o.get('failure_regime') or '?',
            bool(o.get('is_abstention', False)),
        )
    return out

def diff_tables(base, trans, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    common = set(base) & set(trans)
    print(f"CONFIRM questions compared: {len(common)} "
          f"(baseline n={len(base)}, transform n={len(trans)})")

    # Flip table
    w2r = [q for q in common if not base[q][0] and trans[q][0]]
    r2w = [q for q in common if base[q][0] and not trans[q][0]]
    print(f"\n[FLIP TABLE] wrong->right: {len(w2r)} | right->wrong: {len(r2w)} | net: {len(w2r)-len(r2w):+d}")

    # Per-type delta
    print("\n[PER-TYPE DELTA] (CONFIRM)")
    print(f"{'type':<28}{'base_correct':>13}{'trans_correct':>15}{'delta':>8}")
    types = sorted({base[q][1] for q in common})
    for t in types:
        qs = [q for q in common if base[q][1] == t]
        bc = sum(base[q][0] for q in qs)
        tc = sum(trans[q][0] for q in qs)
        print(f"{t:<28}{bc:>13}{tc:>15}{tc-bc:>+8d}")

    # Abstention
    abst = [q for q in common if base[q][3] or trans[q][3]]
    if abst:
        bc = sum(base[q][0] for q in abst); tc = sum(trans[q][0] for q in abst)
        print(f"\n[ABSTENTION] n={len(abst)} | base_correct={bc} trans_correct={tc} delta={tc-bc:+d}")
    else:
        print("\n[ABSTENTION] no abstention-flagged questions found in CONFIRM (check field name)")

    # Localization: where did the net flips land?
    print("\n[LOCALIZATION] flips by regime")
    flip_by_regime = defaultdict(lambda: [0,0])
    for q in w2r: flip_by_regime[base[q][2]][0]+=1
    for q in r2w: flip_by_regime[base[q][2]][1]+=1
    for reg,(up,down) in sorted(flip_by_regime.items()):
        print(f"  {reg:<28} +{up} / -{down}  (net {up-down:+d})")

def main(path):
    r = load(path)
    keys, promoted = find_confirm_blocks(r)
    print(f"\n##### {path}")
    print("top-level report keys:", keys)
    print("promoted reader-prompt-transforms found:", len(promoted))
    if promoted:
        t = promoted[0]
        print("promoted transform keys:", list(t.keys()))
        print("confirm_delta:", t.get('confirm_delta'), "| overall_delta:", t.get('overall_delta'),
              "| target_delta:", t.get('target_delta'))
    # The script needs the per-question CONFIRM outcomes for baseline and transform.
    # Print where they might live so we can wire the exact path:
    print("\nProbing for per-question outcome arrays...")
    for k,v in r.items():
        if isinstance(v, dict):
            for kk,vv in v.items():
                if isinstance(vv, list) and vv and isinstance(vv[0], dict) and ('question_id' in vv[0] or 'qid' in vv[0] or 'correct' in vv[0]):
                    print(f"  candidate outcomes at: {k}.{kk}  (n={len(vv)}, sample keys={list(vv[0].keys())[:6]})")
        if isinstance(v, list) and v and isinstance(v[0], dict) and ('question_id' in v[0] or 'correct' in v[0]):
            print(f"  candidate outcomes at: {k}  (n={len(v)}, sample keys={list(v[0].keys())[:6]})")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
