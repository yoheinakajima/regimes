"""SqlOutcome — per-question record for the SQL target.

Mirrors the role of `regimes.eval.types.Outcome` for LongMemEval, but
populated with SQL-shaped fields the SQL regime detectors read.

Why subclass `Outcome`? Per the Phase 1 investigation map
(docs/PLATFORM_INVESTIGATION.md §Outcome): "the loop's per-outcome
touch points are narrow enough that subclass-by-convention works". A
couple of loop-control code paths (attribute.py and gates.eval_diff)
still bypass `target.taxonomy.classify` and import LongMemEval's
`classify` directly — see the Phase 1.5 leak notes in
docs/PHASE1_5_LEAKS.md. Subclassing `Outcome` means SqlOutcome already
carries the empty defaults for the LME-shaped fields (`scores={}`,
`answer_session_ids=()`, etc.), so the leaked LME classify() returns
False on every detector and lands on `unclassified` rather than
crashing on a missing attribute. The SQL taxonomy operates on the
SQL-specific fields via `SqlTaxonomy.classify`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regimes.eval.types import Outcome


@dataclass(frozen=True)
class SqlOutcome(Outcome):
    """SQL-specific outcome fields. All have defaults so SqlOutcome
    inherits Outcome's required-positional fields (`question_id`,
    `question_type`, `is_abstention`, `answer_session_ids`, `correct`)
    and adds its own on top."""

    # The natural-language question + schema id (for audit).
    nl_question: str = ""
    schema_id: str = ""

    # What the SQL agent drafted.
    predicted_sql: str = ""
    gold_sql: str = ""

    # Execution outcome on the in-memory sqlite.
    exec_error: str = ""                          # "" iff query ran cleanly
    predicted_result_set: tuple[tuple, ...] = ()
    gold_result_set: tuple[tuple, ...] = ()

    # Schema shape (what the gold answer touches + what the predicted
    # answer touches). Used by detect_schema_misunderstanding,
    # detect_wrong_join, detect_wrong_aggregation, detect_wrong_filter.
    schema_tables: tuple[str, ...] = ()
    schema_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    foreign_keys: tuple[tuple[str, str, str, str], ...] = ()  # (table, col, ref_table, ref_col)

    predicted_tables: tuple[str, ...] = ()
    predicted_columns: tuple[tuple[str, str], ...] = ()  # qualified (table, col) when present
    predicted_has_join: bool = False
    predicted_has_where: bool = False
    predicted_has_group_by: bool = False
    predicted_has_having: bool = False

    gold_tables: tuple[str, ...] = ()
    gold_has_join: bool = False
    gold_has_where: bool = False
    gold_has_group_by: bool = False
    gold_has_having: bool = False

    # Agent-side artifacts (used to build sandbox probes for prompt
    # transforms and to record what the loop's seam saw).
    prompt_parts: dict[str, Any] = field(default_factory=dict)
    schema_meta: dict[str, Any] = field(default_factory=dict)
    selected_column_ids: tuple[str, ...] = ()
    column_scores: dict[str, float] = field(default_factory=dict)
