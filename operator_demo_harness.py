#!/usr/bin/env python3
"""Operator demonstration harness (Regimes Section 8.3).
Tests guarded count_entities_across_sessions vs the prose transform on seed-7 held-out."""
import argparse, json, re

COUNTING_INTENT = re.compile(r'\b(how many|number of|how often|times|count)\b', re.I)
def carries_counting_intent(q): return bool(COUNTING_INTENT.search(q or ""))
def guard_fires(q, sessions_with_match): return carries_counting_intent(q) and sessions_with_match >= 2

def load_confirm(report_path):
    r = json.load(open(report_path))
    conf = r.get('confirm') or (r.get('promotions', [{}])[-1] if r.get('promotions') else {})
    def pick(d, *names):
        for n in names:
            if n in d: return d[n]
        return None
    base = pick(conf, 'confirm_baseline_outcomes', 'baseline_outcomes')
    tran = pick(conf, 'confirm_transform_outcomes', 'transform_outcomes')
    if base is None or tran is None:
        print("[!] Could not find confirm outcome keys. Report top-level keys:")
        print("   ", list(r.keys()))
        if conf: print("    confirm/promotion keys:", list(conf.keys()))
        raise SystemExit("Adjust load_confirm() key names to match your report.json, then rerun.")
    return ({o['question_id']: o for o in base}, {o['question_id']: o for o in tran})

def flips(a, b):
    common = set(a) & set(b)
    return ([q for q in common if not a[q] and b[q]], [q for q in common if a[q] and not b[q]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True); ap.add_argument('--lme-data', required=True)
    ap.add_argument('--target-qid', default='618f13b2')
    a = ap.parse_args()
    base, tran = load_confirm(a.report)
    base_c = {q: bool(o['correct']) for q,o in base.items()}
    tran_c = {q: bool(o['correct']) for q,o in tran.items()}
    qs = {q: o.get('question','') for q,o in base.items()}
    print("=== seed-7 held-out: baseline vs prose transform (committed) ===")
    wr, rw = flips(base_c, tran_c)
    print(f"  baseline acc {sum(base_c.values())/len(base_c):.3f}  transform acc {sum(tran_c.values())/len(tran_c):.3f}")
    print(f"  prose flips: w->r {len(wr)}  r->w {len(rw)}  (regressions: {sorted(rw)})")
    print(f"  target {a.target_qid}: baseline correct={base_c.get(a.target_qid)}, prose correct={tran_c.get(a.target_qid)}")
    tq = qs.get(a.target_qid, '')
    print(f"\n  guard sanity check on target question:\n    {tq!r}\n    counting_intent={carries_counting_intent(tq)}")
    print("\n[OPERATOR CONDITION NOT WIRED]")
    print("  To get the operator result, implement run_operator_condition() to run the reader")
    print("  with count_entities_across_sessions installed over the same held-out qids,")
    print("  scored by the LongMemEval judge, then compute flips(tran_c, oper_c) and flips(base_c, oper_c).")
    print("  Want: target fixed (correct=True), zero new breaks vs baseline.")

if __name__ == '__main__': main()
