"""Tiny regex-based SQL parser.

We only need to answer four questions about a SELECT statement to
classify failures:

  - Which tables are referenced (after FROM / JOIN)?
  - Which qualified columns are referenced (table.column)?
  - Does the statement contain a JOIN clause?
  - Does it have WHERE / GROUP BY / HAVING clauses?

A full SQL parser is overkill — the synthetic fixture's queries are
single-statement SELECTs. Regex over a lowercased, whitespace-normalized
form is enough. For production this would be replaced with sqlparse or
a proper grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_WS = re.compile(r"\s+")
_FROM_OR_JOIN = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_QUALIFIED_COL = re.compile(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedSql:
    tables: tuple[str, ...]
    qualified_columns: tuple[tuple[str, str], ...]
    has_join: bool
    has_where: bool
    has_group_by: bool
    has_having: bool


def parse_sql(sql: str) -> ParsedSql:
    """Best-effort structural fields. Lowercases; tolerates trailing
    semicolons and excess whitespace."""
    if not sql:
        return ParsedSql((), (), False, False, False, False)
    s = _WS.sub(" ", sql.strip().rstrip(";")).lower()

    tables_seen: list[str] = []
    seen_t: set[str] = set()
    for m in _FROM_OR_JOIN.finditer(s):
        t = m.group(1)
        if t not in seen_t:
            seen_t.add(t)
            tables_seen.append(t)

    cols_seen: list[tuple[str, str]] = []
    seen_c: set[tuple[str, str]] = set()
    for m in _QUALIFIED_COL.finditer(s):
        pair = (m.group(1), m.group(2))
        if pair not in seen_c:
            seen_c.add(pair)
            cols_seen.append(pair)

    return ParsedSql(
        tables=tuple(tables_seen),
        qualified_columns=tuple(cols_seen),
        has_join=" join " in f" {s} ",
        has_where=" where " in f" {s} ",
        has_group_by=" group by " in f" {s} ",
        has_having=" having " in f" {s} ",
    )
