"""Tokenization for the lexical signal.

The tokenizer is intentionally tiny: alphanumeric runs, lowercased, then
filtered by min_token_length and the pinned stoplist. Per-turn distinct
tokens are deduped in input order.

For context-token counting (used by the budget walk), we use the
char-divided-by-4 fallback — the LME reference accepts this as
`context_token_source = "charfallback"`. tiktoken can be wired in later
without changing the budget logic; it's a pure swap of `count_tokens`.
"""

from __future__ import annotations

import re

from regimes.agent.stoplist import STOPLIST

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def raw_tokenize(text: str) -> list[str]:
    """All alphanumeric runs, lowercased. No filtering."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def distinctive_tokens(text: str, min_token_length: int) -> list[str]:
    """Tokens that pass the length+stoplist filter, deduped in input order."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokenize(text):
        if len(tok) < min_token_length:
            continue
        if tok in STOPLIST:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def count_tokens(text: str) -> int:
    """Char/4 fallback. Matches LME's `charfallback` mode. Deterministic."""
    return max(1, len(text) // 4)


def render_turn(session_id: str, session_date: str, role: str, content: str) -> str:
    """Standard turn rendering. Identical to the LME reference spec so the
    text bytes embedded in `turn.data["text"]` and counted by `count_tokens`
    match the reference build."""
    return f"[Session {session_id} ({session_date})] {role}: {content}"
