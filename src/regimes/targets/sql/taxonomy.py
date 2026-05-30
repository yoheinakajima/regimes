"""SQL regime taxonomy + deterministic detectors.

Same structural pattern as `regimes.loop.regimes` (the LongMemEval
taxonomy): a fixed list of named regimes, each with a binary detector
function over `SqlOutcome`, plus optimizable / seam-reachable flags
and a priority order. Implements `regimes.target.RegimeTaxonomy`.

Action space: prompt-transforms can re-shape what the LLM sees
(schema text, instructions, hints, question phrasing). They CANNOT
change the LLM itself, the executor, or the gold answer. So:

  Seam-reachable (prompt-fixable):
    schema-misunderstanding  — drafted SQL references non-existent or
                               wrong tables/columns. Prompt can clarify
                               which columns exist.
    wrong-join               — drafted SQL has the wrong JOIN topology
                               vs. gold. Prompt can hint at FK paths.
    wrong-aggregation        — drafted SQL is missing or extra GROUP BY
                               relative to gold. Prompt can hint when
                               to group.
    wrong-filter             — drafted SQL has the wrong WHERE / HAVING
                               shape. Prompt can hint at filter
                               specificity.

  Seam-unreachable (walls):
    syntax-error             — SQL didn't parse / executor raised. Bug
                               in the LLM's output formatting; prompt-
                               level hints rarely fix syntax stability.
    executable-but-wrong     — SQL ran cleanly with the right structure
                               but wrong rows. Reasoning failure
                               downstream of prompt formatting — needs
                               a different model / strategy.
    unclassified             — catch-all.

Priority: syntax-error is checked first (without parseable SQL the
structural detectors are noise); then schema-misunderstanding (most
specific structural mistake); then wrong-aggregation /
wrong-join / wrong-filter (each is a clean structural mismatch);
then executable-but-wrong (the wall for rows-wrong-but-shape-right);
then unclassified."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from regimes.loop.regimes import Regime, HistogramRow
from regimes.targets.sql.outcome import SqlOutcome


# ---------------------------------------------------------------------------
# Detector helpers
# ---------------------------------------------------------------------------


def _all_schema_columns(o: SqlOutcome) -> set[tuple[str, str]]:
    """Set of valid (table, column) pairs from the instance's schema."""
    out: set[tuple[str, str]] = set()
    for t, cols in o.schema_columns.items():
        for c in cols:
            out.add((t, c))
    return out


def _predicted_invalid_refs(o: SqlOutcome) -> tuple[set[str], set[tuple[str, str]]]:
    """(unknown_tables, unknown_qualified_columns) — references on the
    predicted SQL that have no match in the instance's schema."""
    schema_tables = set(o.schema_tables)
    schema_cols = _all_schema_columns(o)
    bad_tables = set(o.predicted_tables) - schema_tables
    bad_cols = set(o.predicted_columns) - schema_cols
    return bad_tables, bad_cols


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def detect_syntax_error(o: SqlOutcome) -> bool:
    """The executor raised a syntax/parse error on the drafted SQL.

    sqlite3 reports parse errors as `OperationalError: near "X":
    syntax error` and similar. We also lump empty/null SQL into this
    bucket — if the drafter returned nothing, there's nothing to fix
    at the prompt level."""
    if not o.predicted_sql.strip():
        return True
    if not o.exec_error:
        return False
    msg = o.exec_error.lower()
    return any(
        marker in msg for marker in (
            "syntax error", "near ", "incomplete input", "unrecognized token",
        )
    )


def detect_schema_misunderstanding(o: SqlOutcome) -> bool:
    """Drafted SQL references tables or qualified columns that don't
    exist in the schema. Catches the very common LLM mistake of
    inventing column names or guessing a plural table name.

    Two signals are checked:
      (a) Structural — predicted_tables / predicted_columns include
          names not present in the schema.
      (b) Executor — sqlite raised "no such table/column" at runtime.
          This catches the unqualified-bad-column case our regex
          parser can't see (e.g. `SELECT title FROM products;` where
          'title' is unqualified and 'products' is a valid table).
    """
    if o.correct:
        return False
    if not o.predicted_sql.strip():
        return False
    bad_tables, bad_cols = _predicted_invalid_refs(o)
    if bad_tables or bad_cols:
        return True
    if o.exec_error:
        msg = o.exec_error.lower()
        if "no such column" in msg or "no such table" in msg:
            return True
    return False


def detect_wrong_aggregation(o: SqlOutcome) -> bool:
    """Gold uses GROUP BY but predicted doesn't, or vice versa. A clean
    structural mismatch the prompt can address with a "use GROUP BY
    when aggregating" hint."""
    if o.correct:
        return False
    if not o.predicted_sql.strip():
        return False
    return o.predicted_has_group_by != o.gold_has_group_by


def detect_wrong_join(o: SqlOutcome) -> bool:
    """Gold needs a JOIN but predicted doesn't (or vice versa). Or both
    have JOINs but predicted touches a different set of tables than
    gold."""
    if o.correct:
        return False
    if not o.predicted_sql.strip():
        return False
    if o.predicted_has_join != o.gold_has_join:
        return True
    if o.predicted_has_join and o.gold_has_join:
        return set(o.predicted_tables) != set(o.gold_tables)
    return False


def detect_wrong_filter(o: SqlOutcome) -> bool:
    """Gold uses WHERE or HAVING and predicted doesn't (or vice versa).
    The predicate value can still be wrong even if both have a WHERE;
    that case falls through to executable-but-wrong."""
    if o.correct:
        return False
    if not o.predicted_sql.strip():
        return False
    if o.predicted_has_where != o.gold_has_where:
        return True
    if o.predicted_has_having != o.gold_has_having:
        return True
    return False


def detect_executable_but_wrong(o: SqlOutcome) -> bool:
    """SQL ran cleanly (no exec_error), has the right structural shape
    (tables/joins/groupby/filters match gold) but produced different
    rows than gold. The wall: the LLM's REASONING is wrong, not its
    structural understanding — prompt hints can't fix this."""
    if o.correct:
        return False
    if o.exec_error:
        return False
    if not o.predicted_sql.strip():
        return False
    # All structural detectors must have been False to reach here under
    # the priority order, but assert it explicitly so this is sound to
    # call standalone too.
    if detect_schema_misunderstanding(o):
        return False
    if detect_wrong_aggregation(o):
        return False
    if detect_wrong_join(o):
        return False
    if detect_wrong_filter(o):
        return False
    return True


def detect_unclassified(o: SqlOutcome) -> bool:  # noqa: ARG001
    return True


# ---------------------------------------------------------------------------
# Built-in taxonomy
# ---------------------------------------------------------------------------


_BUILTIN: list[Regime] = [
    Regime(
        name="syntax-error",
        detector=detect_syntax_error,
        optimizable=False,
        seam_reachable=False,
        description=(
            "Drafter returned empty SQL or sqlite reported a parse error. "
            "Prompt-transforms cannot reliably fix output-formatting bugs."
        ),
    ),
    Regime(
        name="schema-misunderstanding",
        detector=detect_schema_misunderstanding,
        optimizable=True,
        seam_reachable=True,
        description=(
            "Predicted SQL references tables or qualified columns absent "
            "from the schema. A prompt-transform that clarifies "
            "available columns can fix it."
        ),
    ),
    Regime(
        name="wrong-aggregation",
        detector=detect_wrong_aggregation,
        optimizable=True,
        seam_reachable=True,
        description=(
            "Gold uses GROUP BY but predicted doesn't (or vice versa). "
            "Promptable via a 'use GROUP BY when aggregating' hint."
        ),
    ),
    Regime(
        name="wrong-join",
        detector=detect_wrong_join,
        optimizable=True,
        seam_reachable=True,
        description=(
            "Predicted SQL joins the wrong tables, or omits a needed "
            "JOIN (or invents one). Promptable via explicit FK-path hints."
        ),
    ),
    Regime(
        name="wrong-filter",
        detector=detect_wrong_filter,
        optimizable=True,
        seam_reachable=True,
        description=(
            "Predicted SQL is missing or has-extra WHERE/HAVING relative "
            "to gold. Promptable via a 'remember to filter rows' hint."
        ),
    ),
    Regime(
        name="executable-but-wrong",
        detector=detect_executable_but_wrong,
        optimizable=False,
        seam_reachable=False,
        description=(
            "SQL ran without error and had the right structural shape "
            "but produced different rows than gold — reasoning failure "
            "downstream of prompt formatting. Wall."
        ),
    ),
    Regime(
        name="unclassified",
        detector=detect_unclassified,
        optimizable=False,
        seam_reachable=False,
        description="Catch-all for outcomes no other detector matches.",
    ),
]


PRIORITY: tuple[str, ...] = (
    "syntax-error",
    "schema-misunderstanding",
    "wrong-aggregation",
    "wrong-join",
    "wrong-filter",
    "executable-but-wrong",
    "unclassified",
)


# ---------------------------------------------------------------------------
# Taxonomy adapter (implements regimes.target.RegimeTaxonomy)
# ---------------------------------------------------------------------------


@dataclass
class SqlTaxonomy:
    """Per-instance LME-shaped taxonomy state. Unlike the LongMemEval
    taxonomy (which uses a process-global registry), we keep state on
    the instance so multiple SqlTargets can coexist without sharing a
    registry."""

    name: str = "sql"
    _registry: dict[str, Regime] = field(default_factory=dict)
    _priority: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if not self._registry:
            self._registry = {r.name: r for r in _BUILTIN}
            self._priority = list(PRIORITY)

    def REGIMES(self) -> dict[str, Regime]:  # noqa: N802
        with self._lock:
            return dict(self._registry)

    def classify(self, outcome: Any) -> Regime:
        with self._lock:
            for name in self._priority:
                r = self._registry[name]
                if r.detector(outcome):
                    return r
            return self._registry["unclassified"]

    def histogram(self, outcomes: Sequence[Any], *, failures_only: bool = True) -> list[HistogramRow]:
        target = [o for o in outcomes if (not failures_only) or (not o.correct)]
        by_regime: dict[str, list[Any]] = {n: [] for n in self._priority}
        for o in target:
            r = self.classify(o)
            by_regime.setdefault(r.name, []).append(o)
        with self._lock:
            rows = []
            for name in self._priority:
                r = self._registry[name]
                members = by_regime.get(name, [])
                rows.append(HistogramRow(
                    regime=name,
                    count=len(members),
                    optimizable=r.optimizable,
                    seam_reachable=r.seam_reachable,
                    qids=tuple(o.question_id for o in members),
                ))
        return rows

    def is_seam_reachable(self, regime_name: str) -> bool:
        with self._lock:
            r = self._registry.get(regime_name)
        return bool(r and r.seam_reachable)

    def format_histogram(self, rows: Sequence[HistogramRow], *, n_failures: int, n_total: int) -> str:
        lines = [
            f"SQL regime histogram (failures={n_failures} / total={n_total}):",
            f"  {'regime':<26s}  {'count':>5s}  {'opt':>4s}  {'seam':>5s}",
        ]
        for r in rows:
            flag_opt = "yes" if r.optimizable else "no"
            flag_seam = "yes" if r.seam_reachable else "no"
            lines.append(
                f"  {r.regime:<26s}  {r.count:>5d}  {flag_opt:>4s}  {flag_seam:>5s}"
            )
        return "\n".join(lines)

    def name_wall(self, counts: Mapping[str, int]) -> str:
        reg = self.REGIMES()
        fragments: list[str] = []
        for name, c in sorted(counts.items()):
            if c <= 0:
                continue
            r = reg.get(name)
            if r is None or (r.optimizable and r.seam_reachable):
                continue
            if name == "executable-but-wrong":
                fix = "SQL agent reasoning change (different LLM, finetune, or different prompt strategy)"
            elif name == "syntax-error":
                fix = "fix the drafter's output formatting (model upgrade or post-processing)"
            else:
                fix = "outside the prompt-transform action space"
            fragments.append(f"{name}={c} → {fix}")
        return "; ".join(fragments) if fragments else "no remaining failures"

    def register_regime(
        self,
        name: str,
        detector: Callable[[Any], bool],
        *,
        optimizable: bool,
        seam_reachable: bool,
        description: str = "",
        priority_after: str = "wrong-filter",
    ) -> None:
        """LLM-proposed regime hook. Same shape as LME's
        `register_regime` but lives on the SqlTaxonomy instance."""
        with self._lock:
            if name in self._registry:
                raise ValueError(f"regime already registered: {name!r}")
            self._registry[name] = Regime(
                name=name, detector=detector,
                optimizable=optimizable, seam_reachable=seam_reachable,
                description=description,
            )
            try:
                idx = self._priority.index(priority_after)
            except ValueError:
                idx = len(self._priority) - 1
            self._priority.insert(idx + 1, name)
