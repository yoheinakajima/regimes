"""Transform authoring.

`hypothesize` consumes the regime histogram and produces a candidate
score-transform: a name + source string targeting the dominant
optimizable regime. Authoring is INERT — `transform.drafted` is emitted
and nothing else changes. The static-analysis gate runs next.

Two authors are provided:

  StubAuthor — deterministic. Picks from a small library of
               pre-written transforms keyed by target regime. Used for
               every test and for the no-keys MockEval run.

  LLMAuthor  — calls an Anthropic Claude model with the failing
               outcomes + targeted regime + a transform-signature hint
               and returns the model's source string. Used only on the
               real-eval path. Construction validates ANTHROPIC_API_KEY
               + the `anthropic` import; missing either is a
               ConfigurationError (caller-fixable).

Both authors return a `DraftedTransform` — a plain dataclass the
behaviors emit into the event log. The promotion gate later compiles
the source string in-place to a callable; we do NOT carry callables in
event payloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from activegraph import ConfigurationError

from regimes.eval.types import Outcome


@dataclass(frozen=True)
class DraftedTransform:
    name: str
    source: str
    target_regime: str
    author: str          # "stub" | "claude-sonnet-4-5"
    rationale: str = ""


# ---------------------------------------------------------------------------
# The transform contract reminder, kept here so authors emit code that
# matches what the seam expects. Both stub source strings and LLM prompts
# are constructed around this signature.
# ---------------------------------------------------------------------------

TRANSFORM_SIGNATURE = (
    "def transform(scores: dict, graph, question: str, question_date: str) -> dict:"
)


# ---------------------------------------------------------------------------
# StubAuthor — a small library of pre-written transforms per regime.
# Source strings are deliberately tiny and AST-clean (math only,
# no imports) so they pass the static gate.
# ---------------------------------------------------------------------------


_STUB_LIBRARY: dict[str, tuple[str, str, str]] = {
    # target_regime: (suggested_name, source, rationale)
    "assembly-crowding": (
        "stub_topk_boost",
        (
            "def transform(scores, graph, question, question_date):\n"
            "    if not scores:\n"
            "        return scores\n"
            "    items = sorted(scores.items(), key=lambda kv: -kv[1])\n"
            "    out = {}\n"
            "    for i, (tid, s) in enumerate(items):\n"
            "        if i < 5:\n"
            "            out[tid] = s * 1.25\n"
            "        else:\n"
            "            out[tid] = s\n"
            "    return out\n"
        ),
        "Boost the top-5-scored turns by 25% to pull them past the seed "
        "threshold before lower-ranked turns crowd them out.",
    ),
    "budget-truncation": (
        "stub_demote_low",
        (
            "def transform(scores, graph, question, question_date):\n"
            "    if not scores:\n"
            "        return scores\n"
            "    vals = sorted(scores.values())\n"
            "    if not vals:\n"
            "        return scores\n"
            "    cutoff = vals[len(vals) // 2]\n"
            "    out = {}\n"
            "    for tid, s in scores.items():\n"
            "        out[tid] = s if s >= cutoff else s * 0.5\n"
            "    return out\n"
        ),
        "Halve below-median scores so bottom-half filler doesn't eat "
        "budget that gold turns need to be included.",
    ),
}

# Order in which StubAuthor picks a target. Both regimes here are
# optimizable + seam-reachable.
_TARGET_PRIORITY: tuple[str, ...] = ("budget-truncation", "assembly-crowding")


@dataclass
class StubAuthor:
    name: str = "stub"

    def draft(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Outcome],   # noqa: ARG002 — kept for parity with LLMAuthor
    ) -> DraftedTransform:
        if dominant_regime in _STUB_LIBRARY:
            n, src, rat = _STUB_LIBRARY[dominant_regime]
            return DraftedTransform(
                name=n, source=src, target_regime=dominant_regime,
                author=self.name, rationale=rat,
            )
        # Fall through to whatever regime IS in the library.
        for r in _TARGET_PRIORITY:
            n, src, rat = _STUB_LIBRARY[r]
            return DraftedTransform(
                name=n, source=src, target_regime=r,
                author=self.name, rationale=rat,
            )
        # Defensive — _STUB_LIBRARY is never empty in practice.
        raise RuntimeError("StubAuthor has no library entries")  # pragma: no cover

    def pick_target(self, regime_counts: dict[str, int]) -> str:
        """Choose the highest-count optimizable+seam-reachable regime."""
        for r in _TARGET_PRIORITY:
            if regime_counts.get(r, 0) > 0:
                return r
        return ""


# ---------------------------------------------------------------------------
# LLMAuthor — Claude-backed authoring. Construction validates env + import.
# ---------------------------------------------------------------------------


@dataclass
class LLMAuthor:
    """Real authoring path. Not exercised in the in-container test suite
    (no API key); covered by integration tests on the user's machine."""

    name: str = "claude-sonnet-4-5"
    temperature: float = 0.2
    max_tokens: int = 1024
    _client: object | None = None

    def __post_init__(self) -> None:
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ConfigurationError(
                "LLMAuthor requires ANTHROPIC_API_KEY in the environment."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ConfigurationError(
                "LLMAuthor requires the `anthropic` package. "
                "Install: pip install regimes[eval]"
            ) from e

    def _ensure_client(self):  # pragma: no cover — network path
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def draft(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Outcome],
    ) -> DraftedTransform:  # pragma: no cover — network path
        cli = self._ensure_client()
        sample = list(failures)[:5]
        sample_str = "\n".join(
            f"- {o.question_id} ({o.question_type}): truncated={o.truncated}, "
            f"selected={len(o.selected_turn_ids)}, gold_top5="
            f"{bool(o.gold_ranked_top_k(5))}"
            for o in sample
        )
        prompt = (
            f"You are authoring a score-transform to address the "
            f"'{dominant_regime}' regime.\n\n"
            f"Signature (REQUIRED, exact):\n  {TRANSFORM_SIGNATURE}\n\n"
            "Constraints:\n"
            "  - Pure Python; ONLY the `math` module may be imported.\n"
            "  - No filesystem, network, subprocess, or attribute access "
            "on builtins.\n"
            "  - Return a dict over the SAME turn_ids as the input scores.\n\n"
            f"Failing outcomes (sample):\n{sample_str}\n\n"
            "Reply with a single ```python``` block containing only the "
            "function. No prose."
        )
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
        return DraftedTransform(
            name=f"llm_{dominant_regime.replace('-', '_')}",
            source=src,
            target_regime=dominant_regime,
            author=self.name,
            rationale=text[:200],
        )


def _extract_code(text: str) -> str:  # pragma: no cover — network path
    """Pull the first ```python ...``` block; fall back to the raw text."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    for p in parts:
        if p.startswith("python"):
            return p[len("python"):].strip()
        if p.strip().startswith("def transform"):
            return p.strip()
    return text.strip()
