"""Attribution via structural diff over two eval runs.

attribute(before, after) returns the set of questions whose regime
changed under the transform, joining the per-question regime
classifications from the two EvalResults. The runtime's `fork()` is
the ideal mechanism for this when the loop runs against the
activegraph SQLite store: fork at the BASELINE event, install the
transform, replay forward, then diff. In the in-container path we
don't have a persistent store wired up, so we compose attribution
from the two EvalResult snapshots — same structural answer, same
payload shape, no inference.

The output is intentionally a plain set of transitions (qid, from,
to) rather than a free-text explanation. Attribution is provable, not
narrated.

Target-agnostic: `attribute()` takes a `taxonomy=...` keyword (default
= LongMemEval). The loop's `behavior_attribute` passes
`lctx.target.taxonomy` so attribution.recorded events carry the
target's own regime names. Without a taxonomy thread-through, every
non-LME target's attribution would label failures as "unclassified".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from regimes.eval.types import EvalResult
from regimes.loop.regimes import classify as _lme_classify


@dataclass(frozen=True)
class Attribution:
    transitions: tuple[tuple[str, str, str], ...]
    n_recovered: int    # transitions to "correct" — actual gains
    n_introduced: int   # transitions away from "correct" — regressions
    # Taxonomy-of-origin tag for the regime-name strings inside
    # `transitions`. Empty string means "no tag" (older callers).
    taxonomy_name: str = ""

    @property
    def net_recovered(self) -> int:
        return self.n_recovered - self.n_introduced

    def filtered_by_target(
        self, target_regime: str
    ) -> tuple[tuple[str, str, str], ...]:
        return tuple(t for t in self.transitions if t[1] == target_regime)

    def directed_rows(self) -> tuple[dict[str, Any], ...]:
        """Per-transition rows with the DIRECTION made explicit.

        The bare `(qid, before, after)` triples already carry both
        directions (a right→wrong regression is `(qid, "correct",
        <regime>)`), but a consumer has to know that "correct" is the
        sentinel to read direction out of them. These rows surface it
        directly so held-out flip tables and per-category regression
        counts are reconstructable from the saved report without that
        implicit knowledge:

          status "gained"  — wrong→right (before incorrect, now correct)
          status "lost"    — right→wrong (before correct, now incorrect)
          status "shifted" — wrong→wrong, regime label changed
        """
        rows: list[dict[str, Any]] = []
        for qid, bv, av in self.transitions:
            if av == "correct" and bv != "correct":
                status = "gained"
            elif bv == "correct" and av != "correct":
                status = "lost"
            else:
                status = "shifted"
            rows.append(
                {
                    "question_id": qid,
                    "before": bv,
                    "after": av,
                    "before_correct": bv == "correct",
                    "after_correct": av == "correct",
                    "status": status,
                }
            )
        return tuple(rows)


class _TaxonomyLike:
    """Duck-typed protocol so this module doesn't import
    `regimes.target.RegimeTaxonomy` (which would create a cycle —
    `target` re-exports gate shapes from `regimes.loop.gates` which
    indirectly pulls in this attribute module). The real protocol
    lives in `regimes.target` and is structurally identical."""
    def classify(self, outcome: Any) -> Any: ...
    name: str


def _per_qid_regime(
    result: EvalResult, *, taxonomy: _TaxonomyLike | None,
) -> dict[str, str]:
    classify_fn = taxonomy.classify if taxonomy is not None else _lme_classify
    out: dict[str, str] = {}
    for o in result.outcomes:
        out[o.question_id] = "correct" if o.correct else classify_fn(o).name
    return out


def attribute(
    before: EvalResult,
    after: EvalResult,
    *,
    taxonomy: _TaxonomyLike | None = None,
) -> Attribution:
    """Compute the per-qid regime transitions between two EvalResults.

    `taxonomy=None` falls back to the LongMemEval taxonomy so direct
    callers see no change."""
    b = _per_qid_regime(before, taxonomy=taxonomy)
    a = _per_qid_regime(after, taxonomy=taxonomy)
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
    tax_name = getattr(taxonomy, "name", "") if taxonomy is not None else "longmemeval"
    return Attribution(
        transitions=tuple(transitions),
        n_recovered=n_rec,
        n_introduced=n_intro,
        taxonomy_name=tax_name,
    )
