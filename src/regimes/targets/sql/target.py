"""SqlTarget — the concrete `Target` for the SQL agent.

Bundles the four pieces: a `SqlEvalBackend`, a `SqlActionSpace` (prompt-
transform pipeline + SQL-shaped gates), a `SqlTaxonomy` (deterministic
detectors), and a per-outcome `outcome_summary` projection.

`build_target` is the convenience constructor the SQL CLI uses to turn
`(reader, author)` into a full Target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regimes.target import ActionSpace, EvalBackend, RegimeTaxonomy
from regimes.targets.sql.action_space import SqlActionSpace
from regimes.targets.sql.eval import SqlEvalBackend
from regimes.targets.sql.hypothesize import StubSqlAuthor
from regimes.targets.sql.outcome import SqlOutcome
from regimes.targets.sql.taxonomy import SqlTaxonomy


def outcome_summary(o: SqlOutcome) -> dict[str, Any]:
    """Self-justifying per-question summary for persistence — carries
    the structural signals the SQL detectors used to assign the regime
    label so every label in the persisted report is auditable.

    Mirrors LongMemEval's outcome_summary in role: distilled signals,
    not the full Outcome."""
    return {
        "question_id": o.question_id,
        "question_type": o.question_type,
        "correct": o.correct,
        "schema_id": o.schema_id,
        "predicted_sql": o.predicted_sql,
        "gold_sql": o.gold_sql,
        "exec_error": o.exec_error,
        "predicted_tables": list(o.predicted_tables),
        "predicted_qualified_columns": [list(c) for c in o.predicted_columns],
        "schema_tables": list(o.schema_tables),
        "predicted_has_join": o.predicted_has_join,
        "predicted_has_where": o.predicted_has_where,
        "predicted_has_group_by": o.predicted_has_group_by,
        "gold_has_join": o.gold_has_join,
        "gold_has_where": o.gold_has_where,
        "gold_has_group_by": o.gold_has_group_by,
        "applied_transforms": list(o.applied_transforms),
    }


@dataclass
class SqlTarget:
    """Concrete SQL `Target`. Composition only."""

    eval_backend: EvalBackend
    action_space: ActionSpace = field(default_factory=SqlActionSpace)
    taxonomy: RegimeTaxonomy = field(default_factory=SqlTaxonomy)
    name: str = "sql"

    def outcome_summary(self, outcome: SqlOutcome) -> dict[str, Any]:
        return outcome_summary(outcome)


def build_target(
    *,
    reader: Any,
    author: Any = None,
) -> SqlTarget:
    """Build a SqlTarget from a Reader (Fake or Anthropic) and an
    optional author. The action space and the taxonomy SHARE a
    SqlTaxonomy instance so eval_diff sees the same registry the loop's
    diagnose step does."""
    tax = SqlTaxonomy()
    action_space = SqlActionSpace(
        author=author if author is not None else StubSqlAuthor(),
        taxonomy=tax,
    )
    return SqlTarget(
        eval_backend=SqlEvalBackend(reader=reader),
        action_space=action_space,
        taxonomy=tax,
    )
