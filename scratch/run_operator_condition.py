#!/usr/bin/env python3
"""Section 8.3 operator: select_latest_stated_count_under_supersession (entity-anchored).
Single guarded deterministic operator, specified by general rule. Guard: counting intent
AND not temporal AND not abstention AND the entity's quantity stated in >=2 gold sessions.
Extraction: numbers are read ONLY from user turns that mention the question's salient entity
(content nouns from the question), anchored to avoid unrelated numbers. Action: latest-dated
gold session's last entity-anchored number (supersession). Non-firing keeps baseline."""
import argparse, json, re

WORDNUM = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,
           'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,
           'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20}
COUNT_INTENT = re.compile(r'\b(how many|number of|how often|times|count|total)\b', re.I)
TEMPORAL = re.compile(r'\b(how many days|how many weeks|how many months|days ago|weeks ago|months ago|days did it take|days passed|days had passed|days before|weeks passed|months before|years older)\b', re.I)
NUM_TOKEN = re.compile(r'\b(\d{1,3}(?:,\d{3})*|\d+|' + '|'.join(WORDNUM) + r')\b', re.I)
STOP = set('how many number of times count total do i have my the a an in on at to for and is are '
           'did you we so that now already including last week i have been was were of with'.split())

def to_int(tok):
    tok = tok.lower().replace(',', '')
    return int(tok) if tok.isdigit() else WORDNUM.get(tok)

def normalize_answer(s):
    if s is None: return None
    m = NUM_TOKEN.search(str(s).strip().lower())
    return to_int(m.group(1)) if m else None

def is_abstention(qid, gold):
    g = str(gold).lower()
    return qid.endswith('_abs') or 'not enough' in g or 'did not mention' in g

def question_entity_terms(question):
    # salient content words from the question (proper-ish nouns and distinctive words)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]+", question)
    terms = [w for w in words if w.lower() not in STOP and len(w) > 2]
    return [w.lower() for w in terms]

def turn_mentions_entity(text, terms):
    tl = text.lower()
    # require at least one distinctive entity term present in the turn
    return any(t in tl for t in terms)

def operator_guard(question, qid, gold, n_sessions_with_number):
    if not COUNT_INTENT.search(question): return False
    if TEMPORAL.search(question): return False
    if is_abstention(qid, gold): return False
    return n_sessions_with_number >= 2

def operator_answer(record):
    terms = question_entity_terms(record['question'])
    gold_sids = list(record['answer_session_ids'])
    sid_index = {sid: i for i, sid in enumerate(record['haystack_session_ids'])}
    dates = record.get('haystack_dates') or []
    def sort_key(sid):
        i = sid_index.get(sid, 10**9)
        return (dates[i] if i < len(dates) else '', i)
    gold_sorted = sorted([s for s in gold_sids if s in sid_index], key=sort_key)
    last_val, sessions_with_number = None, 0
    for sid in gold_sorted:
        sess = record['haystack_sessions'][sid_index[sid]]
        vals = []
        for t in sess:
            if t.get('role') != 'user': continue
            c = t.get('content','')
            if not turn_mentions_entity(c, terms): continue
            # take numbers in the same sentence as an entity term, else any in turn
            for sent in re.split(r'(?<=[.!?])\s+', c):
                if turn_mentions_entity(sent, terms):
                    vals += [to_int(m.group(1)) for m in NUM_TOKEN.finditer(sent)
                             if to_int(m.group(1)) is not None]
        if vals:
            sessions_with_number += 1
            last_val = vals[-1]
    return last_val, sessions_with_number

def load_confirm(report_path):
    r = json.load(open(report_path))
    conf = r.get('confirm') or (r.get('promotions', [{}])[-1] if r.get('promotions') else {})
    base = conf.get('confirm_baseline_outcomes') or conf.get('baseline_outcomes')
    tran = conf.get('confirm_transform_outcomes') or conf.get('transform_outcomes')
    return ({o['question_id']: bool(o['correct']) for o in base},
            {o['question_id']: bool(o['correct']) for o in tran})

def flips(a, b):
    common = set(a) & set(b)
    return ([q for q in common if not a[q] and b[q]],
            [q for q in common if a[q] and not b[q]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True); ap.add_argument('--lme-data', required=True)
    a = ap.parse_args()
    base_c, tran_c = load_confirm(a.report)
    data = json.load(open(a.lme_data))
    recs = data if isinstance(data, list) else list(data.values())
    by = {r['question_id']: r for r in recs}
    oper_c = dict(base_c); fired = []
    for qid in base_c:
        r = by.get(qid)
        if not r: continue
        val, nsess = operator_answer(r)
        if operator_guard(r['question'], qid, r['answer'], nsess):
            gold_val = normalize_answer(r['answer'])
            correct = (val is not None and gold_val is not None and val == gold_val)
            oper_c[qid] = correct
            fired.append((qid, val, gold_val, correct, r['question_type']))
    print("=== guard fired on %d of %d held-out questions ===" % (len(fired), len(base_c)))
    for qid, val, gv, correct, qt in sorted(fired):
        print("  %s [%s] operator=%s gold=%s correct=%s" % (qid, qt, val, gv, correct))
    print("\n=== accuracy ===")
    print("  baseline : %.3f" % (sum(base_c.values())/len(base_c)))
    print("  prose    : %.3f" % (sum(tran_c.values())/len(tran_c)))
    print("  operator : %.3f" % (sum(oper_c.values())/len(oper_c)))
    print("\n=== operator vs baseline ===")
    wr, rw = flips(base_c, oper_c)
    print("  w->r %d %s" % (len(wr), sorted(wr)))
    print("  r->w %d %s  <- new breaks" % (len(rw), sorted(rw)))
    print("\n=== target checks ===")
    print("  618f13b2 operator correct =", oper_c.get('618f13b2'))
    print("  gpt4_e414231f dormant, correct =", oper_c.get('gpt4_e414231f'))

if __name__ == "__main__":
    main()
