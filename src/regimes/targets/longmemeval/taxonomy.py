"""LongMemEval regime taxonomy adapter.

Thin wrapper around the existing module-level functions in
`regimes.loop.regimes` so the loop can talk to a `RegimeTaxonomy`
instance via `target.taxonomy.*` without changing the detectors or
their priority order.

The `name_wall` method holds the LongMemEval-specific recommendation
strings (signal change / assemble() refactor / scoring-step fix) that
used to live as a private helper in `regimes.loop.behaviors`."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from regimes.loop.regimes import (
    REGIMES,
    HistogramRow,
    Regime,
    classify,
    format_histogram,
    histogram,
    is_seam_reachable,
)


class LongMemEvalTaxonomy:
    """Implements `regimes.target.RegimeTaxonomy` for the LongMemEval
    score-transform action space.

    All methods delegate to the existing module-level functions in
    `regimes.loop.regimes` so the detector priority and the registered
    regime set remain the single source of truth (including any
    LLM-proposed regimes added at runtime via `register_regime`)."""

    name = "longmemeval"

    def REGIMES(self) -> dict[str, Regime]:    # noqa: N802 — public-API style
        return REGIMES()

    def classify(self, outcome: Any) -> Regime:
        return classify(outcome)

    def histogram(self, outcomes: Sequence[Any]) -> list[HistogramRow]:
        return histogram(list(outcomes))

    def is_seam_reachable(self, regime_name: str) -> bool:
        return is_seam_reachable(regime_name)

    def format_histogram(
        self, rows: Sequence[HistogramRow], *, n_failures: int, n_total: int
    ) -> str:
        return format_histogram(list(rows), n_failures=n_failures, n_total=n_total)

    def name_wall(self, counts: Mapping[str, int]) -> str:
        """Construct the named-wall string for the loop.stopped payload.

        Lists the remaining unreachable regimes and what would be needed
        to address each. Pure description; no recommendation about which
        to pursue."""
        reg = self.REGIMES()
        fragments: list[str] = []
        for name, c in sorted(counts.items()):
            if c <= 0:
                continue
            r = reg.get(name)
            if r is None or (r.optimizable and r.seam_reachable):
                continue
            if name == "retrieval-signal-gap":
                fix = "signal change (better embedder / scorer)"
            elif name == "assemble-internal":
                fix = "assemble() internals change (reader prompt / context format)"
            elif name == "scoring-error":
                fix = "fix the scoring-step exception (e.g. input truncation before embedding)"
            else:
                fix = "outside the score-transform action space"
            fragments.append(f"{name}={c} → {fix}")
        return "; ".join(fragments) if fragments else "no remaining failures"
