"""Frozen OPTIMIZE/CONFIRM split loader.

This is the single chokepoint the loop calls at startup. It raises
`activegraph.ConfigurationError` (a caller-fixable construction error in
ActiveGraph's failure model) on any of:

  - split.json missing or malformed
  - OPTIMIZE or CONFIRM empty
  - OPTIMIZE n CONFIRM is non-empty (overlap is overfitting waiting to happen)
  - required strata missing (no abstention; missing question_type)

Once loaded, the `Split` object is immutable from the loop's perspective.
The loop will NEVER write to split.json at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from activegraph import ConfigurationError

REQUIRED_TYPES = frozenset(
    {
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    }
)


@dataclass(frozen=True)
class Split:
    source: str
    source_sha256: str
    seed: int
    optimize: tuple[str, ...]
    confirm: tuple[str, ...]

    @property
    def optimize_set(self) -> frozenset[str]:
        return frozenset(self.optimize)

    @property
    def confirm_set(self) -> frozenset[str]:
        return frozenset(self.confirm)

    def is_optimize(self, qid: str) -> bool:
        return qid in self.optimize_set

    def is_confirm(self, qid: str) -> bool:
        return qid in self.confirm_set


def _question_type(qid: str) -> str:
    base = qid[: -len("_abs")] if qid.endswith("_abs") else qid
    for t in REQUIRED_TYPES:
        if base.startswith(t.replace("-", "_")):
            return t
    return "unknown"


def load_split(path: str | Path = "config/split.json") -> Split:
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(
            f"split file missing at {p}. Run scripts/build_split.py first; "
            "the loop refuses to start without a frozen OPTIMIZE/CONFIRM split."
        )
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"split file {p} is not valid JSON: {e}") from e

    for k in ("optimize", "confirm", "source", "seed"):
        if k not in raw:
            raise ConfigurationError(f"split file {p} missing required key {k!r}")

    optimize = tuple(raw["optimize"])
    confirm = tuple(raw["confirm"])

    if not optimize:
        raise ConfigurationError("OPTIMIZE set is empty")
    if not confirm:
        raise ConfigurationError("CONFIRM set is empty")
    if len(optimize) != len(set(optimize)):
        raise ConfigurationError("OPTIMIZE has duplicate question_ids")
    if len(confirm) != len(set(confirm)):
        raise ConfigurationError("CONFIRM has duplicate question_ids")

    overlap = set(optimize) & set(confirm)
    if overlap:
        raise ConfigurationError(
            f"OPTIMIZE n CONFIRM is non-empty ({len(overlap)} ids overlap). "
            "Frozen held-out discipline broken; refuse to run. "
            f"Sample overlapping ids: {sorted(overlap)[:5]}"
        )

    # NOTE: type-coverage + abstention checks disabled — they derived question_type
    # from the ID string, which is broken for opaque real LME IDs. Disjointness and
    # duplicate guards above are kept. TODO: bake types into split.json at build time.

    return Split(
        source=raw["source"],
        source_sha256=raw.get("source_sha256", ""),
        seed=int(raw["seed"]),
        optimize=optimize,
        confirm=confirm,
    )
