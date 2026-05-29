"""SQL target — the second concrete `regimes.target.Target`.

A text-to-SQL ActiveGraph agent: schema → column-relevance scoring →
prompt assembly (with the configurable prompt_transforms seam) → LLM
drafts SQL → execute against in-memory sqlite → compare result-set
against gold. The same loop machinery from
`regimes.loop` drives it without modification.

Public surface:
    from regimes.targets.sql import SqlTarget, build_target, FakeSqlReader
"""

from __future__ import annotations

from regimes.targets.sql.action_space import (
    SQL_IMPORT_WHITELIST,
    SQL_SIGNATURE_PARAMS,
    SqlActionSpace,
)
from regimes.targets.sql.eval import FakeSqlReader, SqlEvalBackend
from regimes.targets.sql.hypothesize import (
    LLMSqlAuthor,
    StubSqlAuthor,
    build_real_sql_author,
)
from regimes.targets.sql.outcome import SqlOutcome
from regimes.targets.sql.target import SqlTarget, build_target, outcome_summary
from regimes.targets.sql.taxonomy import SqlTaxonomy

__all__ = [
    "FakeSqlReader",
    "LLMSqlAuthor",
    "SQL_IMPORT_WHITELIST",
    "SQL_SIGNATURE_PARAMS",
    "SqlActionSpace",
    "SqlEvalBackend",
    "SqlOutcome",
    "SqlTarget",
    "SqlTaxonomy",
    "StubSqlAuthor",
    "build_real_sql_author",
    "build_target",
    "outcome_summary",
]
