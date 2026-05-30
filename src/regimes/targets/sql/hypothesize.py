"""SQL transform authoring.

`StubSqlAuthor` — deterministic. Picks from a small library of
pre-written prompt-transforms keyed by target regime. The library
sources use *only* the literal whitelist allowed by the static gate
(`math` + `string`); no I/O, no graph mutation. Each transform follows
the contract

    def transform(prompt_parts, question, schema_meta) -> dict

with the exact parameter names — same approach LME uses for its
score-transform stubs.

`LLMSqlAuthor` — Claude-backed authoring. Uses the same
`BEHAVIORDRAFTS_MODEL` env var the LME author honors. Not exercised
in the in-container tests (no API key)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from activegraph import ConfigurationError

from regimes.target import DraftedChange


TRANSFORM_SIGNATURE = (
    "def transform(prompt_parts: dict, question: str, schema_meta: dict) -> dict:"
)


# ---------------------------------------------------------------------------
# StubSqlAuthor — a small library of pre-written prompt-transforms.
# Each transform injects a regime-specific hint into prompt_parts['hints']
# (a list). The unlock phrases below are what FakeSqlReader looks for to
# flip a question to correct in the mock-mode demo.
# ---------------------------------------------------------------------------


# Unlock phrases the synthetic fixture's FakeSqlReader recognizes. Any
# transform that injects one of these (verbatim) into prompt_parts will
# move the corresponding question(s) from default-wrong to correct.
SCHEMA_UNLOCK = "Use exactly the columns listed in the schema."
AGGREGATION_UNLOCK = "When aggregating multiple groups, include GROUP BY."
JOIN_UNLOCK = "Join across tables using their declared foreign keys."
FILTER_UNLOCK = "Apply WHERE clauses to filter rows when the question constrains values."


_STUB_LIBRARY: dict[str, tuple[str, str, str]] = {
    "schema-misunderstanding": (
        "stub_schema_clarification_hint",
        (
            "def transform(prompt_parts, question, schema_meta):\n"
            "    out = dict(prompt_parts)\n"
            "    hints = list(out.get('hints', []))\n"
            f"    hints.append({SCHEMA_UNLOCK!r})\n"
            "    out['hints'] = hints\n"
            "    return out\n"
        ),
        "Inject a 'use exactly the columns listed' hint so the LLM "
        "stops inventing column names.",
    ),
    "wrong-aggregation": (
        "stub_groupby_hint",
        (
            "def transform(prompt_parts, question, schema_meta):\n"
            "    out = dict(prompt_parts)\n"
            "    hints = list(out.get('hints', []))\n"
            f"    hints.append({AGGREGATION_UNLOCK!r})\n"
            "    out['hints'] = hints\n"
            "    return out\n"
        ),
        "Inject a GROUP BY hint so the LLM remembers to group when "
        "the question asks for per-X counts/sums.",
    ),
    "wrong-join": (
        "stub_fk_join_hint",
        (
            "def transform(prompt_parts, question, schema_meta):\n"
            "    out = dict(prompt_parts)\n"
            "    hints = list(out.get('hints', []))\n"
            f"    hints.append({JOIN_UNLOCK!r})\n"
            "    out['hints'] = hints\n"
            "    return out\n"
        ),
        "Inject an FK-join hint so the LLM joins multi-table questions "
        "on the declared foreign keys.",
    ),
    "wrong-filter": (
        "stub_where_hint",
        (
            "def transform(prompt_parts, question, schema_meta):\n"
            "    out = dict(prompt_parts)\n"
            "    hints = list(out.get('hints', []))\n"
            f"    hints.append({FILTER_UNLOCK!r})\n"
            "    out['hints'] = hints\n"
            "    return out\n"
        ),
        "Inject a WHERE-clause hint so the LLM remembers to filter "
        "when the question constrains rows.",
    ),
}


_TARGET_PRIORITY: tuple[str, ...] = (
    "schema-misunderstanding",
    "wrong-aggregation",
    "wrong-join",
    "wrong-filter",
)


@dataclass
class StubSqlAuthor:
    name: str = "stub"

    def draft(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Any],  # noqa: ARG002 — parity with LLM author
    ) -> DraftedChange:
        if dominant_regime in _STUB_LIBRARY:
            n, src, rat = _STUB_LIBRARY[dominant_regime]
            return DraftedChange(
                name=n, source=src, target_regime=dominant_regime,
                author=self.name, rationale=rat,
            )
        # Fall through to whatever regime IS in the library.
        for r in _TARGET_PRIORITY:
            n, src, rat = _STUB_LIBRARY[r]
            return DraftedChange(
                name=n, source=src, target_regime=r,
                author=self.name, rationale=rat,
            )
        raise RuntimeError("StubSqlAuthor has no library entries")  # pragma: no cover


# ---------------------------------------------------------------------------
# LLMSqlAuthor — Anthropic-backed authoring. Env-var override matches
# the LME author's BEHAVIORDRAFTS_MODEL pattern.
# ---------------------------------------------------------------------------


DEFAULT_LLM_MODEL = "claude-sonnet-4-6"


def build_real_sql_author(model: str | None = None) -> "LLMSqlAuthor":
    name = model or os.environ.get("BEHAVIORDRAFTS_MODEL") or DEFAULT_LLM_MODEL
    return LLMSqlAuthor(name=name)


@dataclass
class LLMSqlAuthor:
    """Real authoring path. Not exercised in the in-container tests."""

    name: str = DEFAULT_LLM_MODEL
    temperature: float = 0.2
    max_tokens: int = 2048
    _client: object | None = None

    def __post_init__(self) -> None:
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ConfigurationError(
                "LLMSqlAuthor requires ANTHROPIC_API_KEY in the environment."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ConfigurationError(
                "LLMSqlAuthor requires the `anthropic` package. "
                "Install: pip install regimes[eval]"
            ) from e

    def _ensure_client(self):  # pragma: no cover
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def draft(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Any],
    ) -> DraftedChange:  # pragma: no cover — network path
        cli = self._ensure_client()
        prompt = _build_author_prompt(dominant_regime, list(failures))
        resp = cli.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        src = _extract_code(text)
        return DraftedChange(
            name=f"llm_{dominant_regime.replace('-', '_')}",
            source=src,
            target_regime=dominant_regime,
            author=self.name,
            rationale=text[:200],
        )


def _build_author_prompt(dominant_regime: str, failures: list[Any]) -> str:
    sample_lines = []
    for o in failures[:5]:
        sample_lines.append(
            f"- qid={o.question_id} type={o.question_type}\n"
            f"  question={o.nl_question!r}\n"
            f"  gold_sql={o.gold_sql!r}\n"
            f"  predicted_sql={o.predicted_sql!r}\n"
            f"  exec_error={o.exec_error!r}\n"
        )
    sample = "\n".join(sample_lines) if sample_lines else "(no failures)"
    return (
        f"You are authoring a Python prompt-transform to address the "
        f"'{dominant_regime}' SQL regime.\n\n"
        f"Signature (REQUIRED, exact):\n  {TRANSFORM_SIGNATURE}\n\n"
        "Constraints:\n"
        "  - Pure Python; ONLY `math` and `string` may be imported.\n"
        "  - No filesystem, network, subprocess, no attribute access on\n"
        "    builtins.\n"
        "  - Return a dict whose keys are a SUBSET of the input keys.\n"
        "  - The available keys are: 'schema', 'instructions', 'hints',\n"
        "    'question'. 'hints' is a list of strings.\n\n"
        f"Failing examples:\n{sample}\n\n"
        "Reply with a single ```python``` block containing only the function."
    )


def _extract_code(text: str) -> str:  # pragma: no cover
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    for p in parts:
        if p.startswith("python"):
            return p[len("python"):].strip()
        if p.strip().startswith("def transform"):
            return p.strip()
    return text.strip()
