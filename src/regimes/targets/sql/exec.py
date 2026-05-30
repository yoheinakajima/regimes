"""SQL executor — runs the drafted query against a fresh in-memory
sqlite seeded from the instance's DDL + INSERT rows.

We open a new `sqlite3.connect(":memory:")` per instance so there's no
cross-question state. The schema_ddl is split on `;` and applied; the
seed_rows are applied; then the predicted SQL is executed. Result
sets come back as tuples of tuples.

Comparison rule (for `result_sets_equal`):
  - Same row multiset (order-insensitive when gold has no ORDER BY).
  - Column-order tolerated: we sort each row before comparing if and
    only if both result sets have the same number of columns AND the
    same multiset of cell values per row when sorted within row.

We catch any sqlite exception and return its message in `exec_error`.
Empty result sets are valid (= 0 rows)."""

from __future__ import annotations

import re
import sqlite3
from typing import Any


def execute_sql(
    *,
    schema_ddl: str,
    seed_rows: list[str],
    query: str,
) -> tuple[tuple[tuple, ...] | None, str]:
    """Run `query` against an in-memory sqlite holding `schema_ddl +
    seed_rows`. Returns (rows | None, exec_error). On error, rows is
    None and exec_error is the sqlite message."""
    conn = sqlite3.connect(":memory:")
    try:
        cur = conn.cursor()
        # Apply schema + seeds first. Errors here are harness errors
        # (fixture bug), not predicted-SQL errors — surface them as
        # exec_error too rather than crashing the eval.
        try:
            for stmt in _split_statements(schema_ddl):
                cur.execute(stmt)
            for stmt in seed_rows:
                cur.execute(stmt)
        except sqlite3.Error as e:
            return None, f"schema-setup: {type(e).__name__}: {e}"

        try:
            cur.execute(query)
            rows = cur.fetchall()
        except sqlite3.Error as e:
            return None, f"{type(e).__name__}: {e}"
        return tuple(tuple(r) for r in rows), ""
    finally:
        conn.close()


_STATEMENT_SPLIT = re.compile(r";\s*(?:\n|$)")


def _split_statements(ddl: str) -> list[str]:
    """Split a multi-statement DDL string on `;` boundaries. Trailing
    empty fragments are dropped. Matches sqlite's lenient parser."""
    out = []
    for part in _STATEMENT_SPLIT.split(ddl.strip()):
        part = part.strip().rstrip(";").strip()
        if part:
            out.append(part)
    return out


def result_sets_equal(
    predicted: tuple[tuple, ...] | None,
    gold: tuple[tuple, ...],
) -> bool:
    """Order-insensitive row-multiset equality. Mismatched column-order
    within a row is tolerated when both sides agree on column count: we
    sort each row's cells before comparison. Cells are coerced to a
    stringified form so int/float/None compare consistently."""
    if predicted is None:
        return False
    if len(predicted) != len(gold):
        return False
    if not gold:
        return True

    p_norm = sorted(_normalize_row(r) for r in predicted)
    g_norm = sorted(_normalize_row(r) for r in gold)
    return p_norm == g_norm


def _normalize_row(row: tuple) -> tuple[str, ...]:
    """Stringify + sort cells within a row so two rows with the same
    values in different column orders compare equal."""
    return tuple(sorted(_cell_str(c) for c in row))


def _cell_str(cell: Any) -> str:
    if cell is None:
        return "<NULL>"
    if isinstance(cell, float):
        # Normalize 1.0 vs 1
        if cell.is_integer():
            return str(int(cell))
        return f"{cell:.6f}".rstrip("0").rstrip(".")
    return str(cell)
