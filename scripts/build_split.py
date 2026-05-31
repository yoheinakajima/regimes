"""Generate config/split.json: frozen OPTIMIZE + CONFIRM partitions.

Inputs:
  - Source dataset JSON in LME shape (list of objects with question_id and
    question_type). Defaults to the synthetic fixture if longmemeval_s_cleaned.json
    is not available.

Selection (deterministic; seed=42):
  - OPTIMIZE = stratified 50-question subset, same algorithm as upstream
    `build_smoke_ids.py` (each base question_type appears, >=1 abstention).
  - CONFIRM  = stratified 100-question subset, disjoint from OPTIMIZE, drawn
    from the same dataset minus OPTIMIZE, same stratification rule.

Output:
  config/split.json
    {
      "source": "<path to source>",
      "source_sha256": "<hex>",
      "n_total": <int>,
      "seed": 42,
      "optimize": [<qid>, ...],   # 50, sorted
      "confirm":  [<qid>, ...],   # 100, sorted
      "stratification": { ... }
    }

The loop refuses to start unless split.json exists, parses, and OPTIMIZE n
CONFIRM is empty. Regenerate by re-running this script against the same
source (byte-identical output).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

SEED = 42
N_OPTIMIZE = 50
N_CONFIRM = 100

REQUIRED_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
}


def _stratum(entry: dict) -> tuple[str, bool]:
    return (entry["question_type"], entry["question_id"].endswith("_abs"))


def _stratified_pick(
    pool: list[dict], n: int, seed: int
) -> tuple[list[str], dict[tuple[str, bool], int]]:
    """Same algorithm as upstream LME build_smoke_ids: floor-then-largest-remainder.

    Each non-empty stratum gets >=1 slot; quota is grown by largest fractional
    remainder until n is reached. Within each stratum, IDs are sorted then
    shuffled with a seeded RNG so output is deterministic.
    """
    if n > len(pool):
        raise ValueError(f"requested n={n} exceeds pool size {len(pool)}")

    by_stratum: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for entry in pool:
        by_stratum[_stratum(entry)].append(entry["question_id"])
    strata = sorted(by_stratum.keys())
    total = sum(len(v) for v in by_stratum.values())

    raw = {s: n * len(by_stratum[s]) / total for s in strata}
    floor = {s: max(1, int(raw[s])) for s in strata}
    if sum(floor.values()) > n:
        raise ValueError(
            f"floor=1 per stratum ({sum(floor.values())}) already exceeds n={n} "
            f"across {len(strata)} strata"
        )
    remainder = n - sum(floor.values())
    # ranks by fractional remainder, stable on stratum identity
    order = sorted(strata, key=lambda s: (-(raw[s] - floor[s]), s))
    quota = dict(floor)
    for s in order:
        if remainder == 0:
            break
        if quota[s] < len(by_stratum[s]):
            quota[s] += 1
            remainder -= 1
    if remainder > 0:
        raise ValueError("could not fill quota; some strata exhausted")

    rng = random.Random(seed)
    selected: list[str] = []
    for s in strata:
        ids = sorted(by_stratum[s])  # canonicalize before shuffle so seed is meaningful
        rng.shuffle(ids)
        selected.extend(ids[: quota[s]])
    selected.sort()
    return selected, dict(quota)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_strata_cover(picked_ids: Iterable[str], all_entries: list[dict]) -> None:
    qid2type = {e["question_id"]: e["question_type"] for e in all_entries}
    picked_types = {qid2type[q] for q in picked_ids}
    missing = REQUIRED_TYPES - picked_types
    if missing:
        raise ValueError(f"split missing required question_types: {sorted(missing)}")
    if not any(q.endswith("_abs") for q in picked_ids):
        raise ValueError("split has zero `_abs` instances; stratifier degenerate")


def build_split(src: Path, dst: Path, *, seed: int = SEED) -> dict:
    """Draw a stratified OPTIMIZE/CONFIRM split from `src` into `dst`.

    `seed` controls the within-stratum shuffle. The default (SEED=42)
    reproduces the committed split byte-for-byte. Passing a different
    seed yields a DIFFERENT held-out draw with the SAME stratification
    rule — used to test robustness across draws rather than just
    repeatability on a fixed split.
    """
    data = json.loads(src.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{src} is not a non-empty JSON list")
    # OPTIMIZE first
    optimize_ids, opt_quota = _stratified_pick(data, N_OPTIMIZE, seed=seed)
    _verify_strata_cover(optimize_ids, data)
    # CONFIRM from the remainder (disjoint by construction)
    remainder_pool = [e for e in data if e["question_id"] not in set(optimize_ids)]
    confirm_ids, conf_quota = _stratified_pick(remainder_pool, N_CONFIRM, seed=seed)
    _verify_strata_cover(confirm_ids, data)
    # invariants
    if set(optimize_ids) & set(confirm_ids):
        raise AssertionError("OPTIMIZE and CONFIRM overlap; bug in builder")
    payload = {
        "source": str(src),
        "source_sha256": _sha256(src),
        "n_total": len(data),
        "seed": seed,
        "n_optimize": len(optimize_ids),
        "n_confirm": len(confirm_ids),
        "optimize": optimize_ids,
        "confirm": confirm_ids,
        "stratification": {
            "rule": "(question_type, is_abs); floor=1; largest-remainder fill; sorted+shuffle(seed) within stratum",
            "required_types": sorted(REQUIRED_TYPES),
            "optimize_quotas": {f"{qt}{'_abs' if a else ''}": c for (qt, a), c in opt_quota.items()},
            "confirm_quotas": {f"{qt}{'_abs' if a else ''}": c for (qt, a), c in conf_quota.items()},
        },
        "discipline": (
            "Transforms are optimized/selected on OPTIMIZE only. The headline "
            "number is reported on CONFIRM only. Reporting an OPTIMIZE-tuned "
            "number as the result would be overfitting."
        ),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Source dataset JSON. Defaults to longmemeval_s_cleaned.json if "
            "present, else fixtures/synthetic_lme.json."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults to config/split.json for the canonical "
            "seed (42); for any other --split-seed the default becomes "
            "config/split.seed<seed>.json so the committed split is never "
            "overwritten by a fresh draw."
        ),
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=SEED,
        help=(
            "Seed for the within-stratum shuffle. Default 42 reproduces the "
            "committed split. A different seed resamples a DIFFERENT held-out "
            "OPTIMIZE/CONFIRM draw with the SAME stratification rule (to test "
            "robustness across draws). With a non-default seed and no explicit "
            "--out, output is written to config/split.seed<seed>.json."
        ),
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = (
            Path("config/split.json")
            if args.split_seed == SEED
            else Path(f"config/split.seed{args.split_seed}.json")
        )

    if args.source is None:
        candidates = [
            Path("data/longmemeval_s_cleaned.json"),
            Path("fixtures/synthetic_lme.json"),
        ]
        for c in candidates:
            if c.exists():
                args.source = c
                break
        if args.source is None:
            print(
                "error: no source dataset found. Run scripts/build_fixture.py "
                "first, or place longmemeval_s_cleaned.json under data/.",
                file=sys.stderr,
            )
            return 2

    report = build_split(args.source, args.out, seed=args.split_seed)
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "source": report["source"],
                "n_total": report["n_total"],
                "seed": report["seed"],
                "n_optimize": report["n_optimize"],
                "n_confirm": report["n_confirm"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
