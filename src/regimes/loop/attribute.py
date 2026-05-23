"""Attribution via structural diff over two eval runs.

attribute(before, after) returns the set of questions whose regime
changed under the transform, joining the per-question regime
classifications from the two EvalResults. The runtime's `fork()` is the
ideal mechanism for this when the loop runs against the activegraph
SQLite store: fork at the BASELINE event, install the transform, replay
forward, then diff. In the in-container path we don't have a persistent
store wired up, so we compose attribution from the two EvalResult
snapshots — same structural answer, same payload shape, no inference.

The output is intentionally a plain set of transitions (qid, from,
to) rather than a free-text explanation. Attribution is provable, not
narrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from regimes.eval.types import EvalResult
from regimes.loop.regimes import classify


@dataclass(frozen=True)
class Attribution:
    transitions: tuple[tuple[str, str, str], ...]
    n_recovered: int    # transitions to "correct" — actual gains
    n_introduced: int   # transitions away from "correct" — regressions

    @property
    def net_recovered(self) -> int:
        return self.n_recovered - self.n_introduced

    def filtered_by_target(
        self, target_regime: str
    ) -> tuple[tuple[str, str, str], ...]:
        return tuple(t for t in self.transitions if t[1] == target_regime)


def _per_qid_regime(result: EvalResult) -> dict[str, str]:
    out: dict[str, str] = {}
    for o in result.outcomes:
        out[o.question_id] = "correct" if o.correct else classify(o).name
    return out


def attribute(before: EvalResult, after: EvalResult) -> Attribution:
    b = _per_qid_regime(before)
    a = _per_qid_regime(after)
    transitions: list[tuple[str, str, str]] = []
    n_rec = 0
    n_intro = 0
    for qid in sorted(set(b) | set(a)):
        bv = b.get(qid, "missing")
        av = a.get(qid, "missing")
        if av == bv:
            continue
        transitions.append((qid, bv, av))
        if av == "correct" and bv != "correct":
            n_rec += 1
        elif bv == "correct" and av != "correct":
            n_intro += 1
    return Attribution(
        transitions=tuple(transitions),
        n_recovered=n_rec,
        n_introduced=n_intro,
    )
