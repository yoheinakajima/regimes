"""Build the synthetic LME-shaped fixture used by MockEval and by the
no-data path of `build_split.py`.

The fixture mirrors `LMEInstance` field shape (question_id, question_type,
question, answer, question_date, haystack_session_ids, haystack_dates,
haystack_sessions, answer_session_ids) so the same split logic and the
same MockEval pipeline work whether we're pointed at this fixture or at
`longmemeval_s_cleaned.json`.

Seed is fixed; rerunning is byte-identical. Commit the output.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from collections import Counter
from pathlib import Path

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)
ABS_FRACTION = 0.10  # ~10% abstention, matches upstream rough proportion
N_DEFAULT = 200
SEED_DEFAULT = 42


def _make_question(rng: random.Random, qid: str, qtype: str) -> dict:
    n_sessions = rng.randint(3, 8)
    session_ids = [f"{qid}_sess{i}" for i in range(n_sessions)]
    dates = [f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" for _ in range(n_sessions)]
    sessions = []
    for _ in range(n_sessions):
        n_turns = rng.randint(2, 6)
        turns = []
        for t in range(n_turns):
            role = "user" if t % 2 == 0 else "assistant"
            content = " ".join(
                "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
                for _ in range(rng.randint(8, 20))
            )
            turns.append({"role": role, "content": content})
        sessions.append(turns)
    # mark one or two sessions as answer-bearing (unless _abs)
    is_abs = qid.endswith("_abs")
    if is_abs:
        answer_session_ids: list[str] = []
        answer = "I don't know."
    else:
        k = rng.randint(1, 2 if qtype == "multi-session" else 1)
        answer_session_ids = rng.sample(session_ids, k=k)
        answer = "synthetic-answer-" + "".join(rng.choices(string.ascii_lowercase, k=12))
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": f"synthetic question for {qid}",
        "answer": answer,
        "question_date": rng.choice(dates),
        "haystack_session_ids": session_ids,
        "haystack_dates": dates,
        "haystack_sessions": sessions,
        "answer_session_ids": answer_session_ids,
    }


def build(n: int, seed: int, dst: Path) -> dict:
    rng = random.Random(seed)
    instances: list[dict] = []
    # Stratify roughly evenly across the six question_types, with a
    # consistent ~10% abstention slice within each.
    per_type = n // len(QUESTION_TYPES)
    remainder = n - per_type * len(QUESTION_TYPES)
    quotas = {qt: per_type for qt in QUESTION_TYPES}
    # spread the remainder
    for qt in list(QUESTION_TYPES)[:remainder]:
        quotas[qt] += 1
    for qt, count in quotas.items():
        n_abs = max(1, int(round(count * ABS_FRACTION)))
        n_pos = count - n_abs
        for i in range(n_pos):
            qid = f"{qt.replace('-', '_')}_q{i:03d}"
            instances.append(_make_question(rng, qid, qt))
        for i in range(n_abs):
            qid = f"{qt.replace('-', '_')}_q{i:03d}_abs"
            instances.append(_make_question(rng, qid, qt))
    instances.sort(key=lambda x: x["question_id"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(instances, indent=2, sort_keys=True) + "\n")
    counts = Counter((i["question_type"], i["question_id"].endswith("_abs")) for i in instances)
    return {"n": len(instances), "counts": {f"{qt}{'_abs' if a else ''}": c for (qt, a), c in counts.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("fixtures/synthetic_lme.json"))
    args = ap.parse_args()
    report = build(args.n, args.seed, args.out)
    print(json.dumps({"wrote": str(args.out), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
