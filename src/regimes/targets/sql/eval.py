"""SqlEvalBackend + readers.

SqlEvalBackend.run_on_split(instances) -> EvalResult:
  - For each instance: drive the SQL agent (which runs prompt_transforms
    if any are promoted), execute the drafted SQL against an in-memory
    sqlite seeded from the fixture, compare result-set against gold,
    construct a SqlOutcome.
  - Return an EvalResult whose `outcomes` is the list of SqlOutcomes.

FakeSqlReader: deterministic. Carries a per-question (gold, default-
wrong, unlock_phrase) tuple. When `unlock_phrase` is present in the
assembled prompt it returns gold; otherwise it returns the default
wrong. The unlock_phrase mechanism is how prompt_transforms move the
needle in mock mode: a promoted transform that injects the right hint
flips the matching questions to correct.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from regimes.eval.types import EvalResult
from regimes.targets.sql import prompt_transforms as _pipeline
from regimes.targets.sql.agent import retrieve as sql_agent_retrieve
from regimes.targets.sql.exec import execute_sql, result_sets_equal
from regimes.targets.sql.outcome import SqlOutcome
from regimes.targets.sql.sql_parse import parse_sql


@dataclass
class FakeSqlReader:
    """Deterministic test reader.

    `table[qid]` = (gold_sql, default_wrong_sql, unlock_phrase_or_None).
    If the assembled prompt contains the unlock phrase (case-sensitive
    substring) we return gold_sql; otherwise we return default_wrong.
    Questions whose default_wrong already equals gold (or whose unlock
    phrase is empty) will simply be correct on baseline."""

    table: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    name: str = "fake-sql-reader-v1"

    def answer(self, *, context: str, question: str, question_id: str) -> str:  # noqa: ARG002
        gold, wrong, unlock = self.table.get(question_id, ("", "", ""))
        if unlock and unlock in context:
            return gold
        return wrong or gold


@dataclass
class SqlEvalBackend:
    """The SQL eval backend. Composes a Reader + the SQL agent + the
    sqlite executor; runs them over a list of instances and returns
    SqlOutcome records ready for diagnose."""

    reader: Any
    name: str = "sql-eval-backend-v1"

    def run_on_split(
        self,
        instances: Iterable[dict[str, Any]],
        *,
        run_dir: str | Path | None = None,
    ) -> EvalResult:
        instances = list(instances)
        outcomes: list[SqlOutcome] = []

        for inst in instances:
            qid = inst["question_id"]
            qtype = inst["question_type"]

            trace = sql_agent_retrieve(inst, reader=self.reader)
            drafted = trace.drafted
            predicted_sql = drafted.predicted_sql

            rows, exec_err = execute_sql(
                schema_ddl=inst["schema_ddl"],
                seed_rows=list(inst["seed_rows"]),
                query=predicted_sql,
            )
            gold_rows = tuple(tuple(r) for r in inst["gold_result_set"])
            correct = result_sets_equal(rows, gold_rows)

            parsed_pred = parse_sql(predicted_sql)
            parsed_gold = parse_sql(inst["gold_sql"])

            outcomes.append(SqlOutcome(
                question_id=qid,
                question_type=qtype,
                is_abstention=False,
                answer_session_ids=(),    # not used by SQL taxonomy
                correct=correct,
                judge_label="sql-1" if correct else "sql-0",
                judge_raw=None,
                hypothesis=predicted_sql,
                run_id=trace.run_id,
                error=(drafted.drafter_error or None),
                score_error="",
                applied_transforms=drafted.applied_transforms,
                # SQL-specific
                nl_question=inst["question"],
                schema_id=inst.get("schema_id", ""),
                predicted_sql=predicted_sql,
                gold_sql=inst["gold_sql"],
                exec_error=exec_err,
                predicted_result_set=(rows or ()),
                gold_result_set=gold_rows,
                schema_tables=tuple(inst["tables"]),
                schema_columns={k: tuple(v) for k, v in inst["columns_by_table"].items()},
                foreign_keys=tuple(tuple(fk) for fk in inst.get("foreign_keys", ())),
                predicted_tables=parsed_pred.tables,
                predicted_columns=parsed_pred.qualified_columns,
                predicted_has_join=parsed_pred.has_join,
                predicted_has_where=parsed_pred.has_where,
                predicted_has_group_by=parsed_pred.has_group_by,
                predicted_has_having=parsed_pred.has_having,
                gold_tables=parsed_gold.tables,
                gold_has_join=parsed_gold.has_join,
                gold_has_where=parsed_gold.has_where,
                gold_has_group_by=parsed_gold.has_group_by,
                gold_has_having=parsed_gold.has_having,
                prompt_parts=dict(drafted.prompt_parts),
                schema_meta=dict(drafted.schema_meta),
                selected_column_ids=drafted.selected_column_ids,
                column_scores=dict(drafted.column_scores),
            ))

        per_type_correct: dict[str, int] = defaultdict(int)
        per_type_total: dict[str, int] = defaultdict(int)
        n_errors = 0
        for o in outcomes:
            per_type_total[o.question_type] += 1
            if o.correct:
                per_type_correct[o.question_type] += 1
            if o.error or o.exec_error:
                n_errors += 1

        aggregate = {
            "version": "regimes-sql-eval-v1",
            "n": len(outcomes),
            "overall_accuracy": (
                sum(1 for o in outcomes if o.correct) / len(outcomes)
                if outcomes else 0.0
            ),
            "per_type_accuracy": {
                t: per_type_correct[t] / per_type_total[t]
                for t in sorted(per_type_total)
            },
            "n_errors": n_errors,
        }

        if run_dir is not None:
            rd = Path(run_dir)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

        return EvalResult(
            outcomes=outcomes,
            aggregate=aggregate,
            backend="sql",
            run_dir=(str(run_dir) if run_dir is not None else None),
            config={
                "reader": getattr(self.reader, "name", ""),
                "applied_transforms": [e.name for e in _pipeline.get_pipeline()],
            },
        )
