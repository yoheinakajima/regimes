"""Build config/sql_split.json — OPTIMIZE / CONFIRM partition for the
SQL fixture.

Mirrors `scripts/build_split.py` for LME: stratified by question_type,
seed=42, deterministic. The fixture is small (30 questions) so
OPTIMIZE is 18 and CONFIRM is 12; that lets the loop's held-out
discipline kick in while leaving real signal in CONFIRM."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "fixtures" / "synthetic_sql.json"
DEFAULT_OUT = REPO / "config" / "sql_split.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optimize-fraction", type=float, default=0.6)
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    data = json.loads(src.read_text())
    sha = hashlib.sha256(src.read_bytes()).hexdigest()

    # Stratify by question_type.
    by_type: dict[str, list[str]] = defaultdict(list)
    for inst in data:
        by_type[inst["question_type"]].append(inst["question_id"])

    rng = random.Random(args.seed)
    optimize: list[str] = []
    confirm: list[str] = []
    for qtype in sorted(by_type):
        qids = sorted(by_type[qtype])
        rng.shuffle(qids)
        cut = max(1, int(round(len(qids) * args.optimize_fraction)))
        optimize.extend(qids[:cut])
        confirm.extend(qids[cut:])

    optimize.sort()
    confirm.sort()
    assert not (set(optimize) & set(confirm)), "OPTIMIZE ∩ CONFIRM must be empty"

    payload = {
        "source": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src),
        "source_sha256": sha,
        "n_total": len(data),
        "seed": args.seed,
        "optimize_fraction": args.optimize_fraction,
        "optimize": optimize,
        "confirm": confirm,
        "stratified_by": "question_type",
        "note": (
            "SQL OPTIMIZE/CONFIRM split. Mirrors the LME held-out "
            "discipline: OPTIMIZE drives diagnose / sandbox / eval-diff; "
            "CONFIRM is touched once per promotion for reporting."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote OPTIMIZE={len(optimize)} / CONFIRM={len(confirm)} → "
        f"{out.relative_to(REPO)}", file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
