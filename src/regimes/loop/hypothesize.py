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
    author: str          # "stub" | "claude-sonnet-4-6" | ...
    rationale: str = ""


# ---------------------------------------------------------------------------
# The transform contract reminder, kept here so authors emit code that
# matches what the seam expects. Both stub source strings and LLM prompts
# are constructed around this signature.
# ---------------------------------------------------------------------------

TRANSFORM_SIGNATURE = (
    "def transform(scores: dict, graph, question: str, question_date: str) -> dict:"
)

# Per-type signatures for the widened LongMemEval action space. The
# author must draft a transform whose signature matches the type the
# action space selected for the diagnosed regime — otherwise the static
# gate rejects it on a signature mismatch. These mirror the
# `signature_params` pinned on each TransformType descriptor in
# `regimes.targets.longmemeval.transform_types`.
SCORE_TRANSFORM_SIGNATURE = TRANSFORM_SIGNATURE
ASSEMBLY_TRANSFORM_SIGNATURE = (
    "def transform(selected_turns: list, scores: dict, "
    "question: str, question_date: str) -> list:"
)
READER_PROMPT_TRANSFORM_SIGNATURE = (
    "def transform(prompt_parts: dict, question: str, question_date: str) -> dict:"
)

# Reader-prompt-transforms may only add a bounded amount of text. Kept in
# sync with the cap the reader-prompt value-validator enforces.
READER_PROMPT_MAX_ADDED_CHARS = 2000


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


DEFAULT_LLM_MODEL = "claude-sonnet-4-6"


def build_real_author(model: str | None = None) -> "LLMAuthor":
    """Construct an LLMAuthor for the --mode real path.

    Reads `BEHAVIORDRAFTS_MODEL` from the environment if `model` is None;
    falls back to `DEFAULT_LLM_MODEL`. Raises `ConfigurationError` (via
    LLMAuthor.__post_init__) if `ANTHROPIC_API_KEY` or the `anthropic`
    package is missing.
    """
    name = model or os.environ.get("BEHAVIORDRAFTS_MODEL") or DEFAULT_LLM_MODEL
    return LLMAuthor(name=name)


@dataclass
class LLMAuthor:
    """Real authoring path. Not exercised in the in-container test suite
    (no API key); covered by integration tests on the user's machine."""

    name: str = DEFAULT_LLM_MODEL
    temperature: float = 0.2
    max_tokens: int = 2048
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
    ) -> DraftedTransform:
        """Legacy single-type path: always drafts a score-transform.

        Kept for callers (and the action space's fallback branch) that
        don't pass an active transform type. Equivalent to
        `draft_typed(..., transform_type="score-transform")` but keeps
        the historical `llm_<regime>` name."""
        return self._draft(
            dominant_regime=dominant_regime,
            failures=failures,
            transform_type="score-transform",
            name=f"llm_{dominant_regime.replace('-', '_')}",
        )

    def draft_typed(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Outcome],
        transform_type: str,
    ) -> DraftedTransform:
        """Draft a transform of the type the action space requested.

        The LongMemEvalActionSpace sets its `_active_type` and passes the
        type name here; the prompt is built so the model emits a function
        whose SIGNATURE and CONSTRAINTS match that type, and so it
        receives the failure signals relevant to that type. Drafting the
        wrong type would fail the per-type static gate."""
        return self._draft(
            dominant_regime=dominant_regime,
            failures=failures,
            transform_type=transform_type,
            name=f"llm_{transform_type.replace('-', '_')}_"
                 f"{dominant_regime.replace('-', '_')}",
        )

    def _draft(
        self,
        *,
        dominant_regime: str,
        failures: Iterable[Outcome],
        transform_type: str,
        name: str,
    ) -> DraftedTransform:
        cli = self._ensure_client()
        sample = list(failures)[:8]
        prompt = build_typed_author_prompt(transform_type, dominant_regime, sample)
        resp = cli.messages.create(  # pragma: no cover — network path
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in resp.content:  # pragma: no cover — network path
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        src = _extract_code(text)
        return DraftedTransform(
            name=name,
            source=src,
            target_regime=dominant_regime,
            author=self.name,
            rationale=text[:200],
        )


def _format_failure_signals(failures: list[Outcome]) -> str:
    """Per-failure evidence signals the author needs to reason about the
    budget-truncation regime: which evidence turns were well-ranked but
    dropped at the budget wall, and the competing turns that won the
    budget."""
    if not failures:
        return "(no failures)"
    blocks: list[str] = []
    for o in failures:
        well_ranked = list(o.evidence_ranked_top_k(5)) \
            if o.has_evidence_turn_ids() else []
        dropped = list(o.evidence_dropped_at_budget()) \
            if o.has_evidence_turn_ids() else []
        selected = list(o.selected_turn_ids)
        # Identify the "budget winners": the non-evidence turns that
        # competed for and won the budget allocation. These are the
        # selected turns that aren't evidence — they're what evictions
        # would target if we down-weight competitors.
        evid = set(o.gold_evidence_turn_ids)
        budget_winners = [t for t in selected if t not in evid]
        # Top-of-rank scores: what the agent saw when it ranked.
        top_ranked = list(o.ranked[:6])
        score_view = {t: round(o.scores.get(t, 0.0), 4) for t in top_ranked}
        evid_scores = {
            t: round(o.scores.get(t, 0.0), 4) for t in o.gold_evidence_turn_ids
        }
        rank_of = o.evidence_rank_positions() if o.has_evidence_turn_ids() else {}
        blocks.append(
            f"- qid={o.question_id} type={o.question_type} "
            f"truncated={o.truncated} n_selected={len(selected)}\n"
            f"    gold_evidence_turn_ids={list(o.gold_evidence_turn_ids)}\n"
            f"    evidence_ranks={rank_of}\n"
            f"    evidence_well_ranked_top5={well_ranked}\n"
            f"    evidence_dropped_at_budget={dropped}\n"
            f"    evidence_scores={evid_scores}\n"
            f"    budget_winners (selected non-evidence)={budget_winners}\n"
            f"    top_ranked_scores={score_view}"
        )
    return "\n".join(blocks)


def build_typed_author_prompt(
    transform_type: str, dominant_regime: str, sample: list[Outcome]
) -> str:
    """Construct the author prompt for the requested transform type.

    Each type gets its own signature, constraints, and failure-signal
    framing. Unknown types fall back to the score-transform prompt."""
    if transform_type == "assembly-transform":
        return _build_assembly_prompt(
            dominant_regime, _format_failure_signals(sample)
        )
    if transform_type == "reader-prompt-transform":
        return _build_reader_prompt(
            dominant_regime, _format_reconciliation_signals(sample)
        )
    # score-transform (default / legacy)
    return _build_author_prompt(dominant_regime, _format_failure_signals(sample))


def _format_reconciliation_signals(failures: list[Outcome]) -> str:
    """Per-failure signals the reader-prompt author needs: the questions
    where the evidence WAS present in the assembled context but the
    reader still answered wrong (reconciliation failures). These are the
    assemble-internal cases a prompt edit can target — the retrieval
    worked, so only the reader's instructions can move them."""
    if not failures:
        return "(no failures)"
    blocks: list[str] = []
    for o in failures:
        # Evidence that actually reached the reader's context window.
        evidence_in_context = list(o.evidence_selected()) \
            if o.has_evidence_turn_ids() else list(o.gold_selected())
        reconciliation_failure = bool(evidence_in_context) and not o.correct
        blocks.append(
            f"- qid={o.question_id} type={o.question_type} "
            f"reconciliation_failure={reconciliation_failure}\n"
            f"    evidence_present_in_context={evidence_in_context}\n"
            f"    n_selected={len(o.selected_turn_ids)}\n"
            f"    judge_label={o.judge_label!r}\n"
            f"    wrong_answer={o.hypothesis[:160]!r}"
        )
    return "\n".join(blocks)


def _build_assembly_prompt(dominant_regime: str, signals_block: str) -> str:
    return (
        f"You are authoring a Python assembly-transform to address the "
        f"'{dominant_regime}' retrieval regime.\n\n"
        f"Signature (REQUIRED, exact):\n  {ASSEMBLY_TRANSFORM_SIGNATURE}\n\n"
        "The transform is called once per question, AFTER assembly has\n"
        "picked `selected_turns` (an ordered list of turn_ids that fit the\n"
        "token budget) and BEFORE the reader sees them. `scores` maps\n"
        "turn_id -> post-scoring weight. Your job is to REORDER and/or\n"
        "FILTER `selected_turns` so that:\n"
        "  - evidence turns that were dropped at the budget wall, or buried\n"
        "    beneath the competing turns that crowded them out, end up\n"
        "    earlier (or survive a filter), AND\n"
        "  - you do NOT invent turn_ids.\n\n"
        "Constraints:\n"
        "  - Pure Python; ONLY the `math` and `string` modules may be\n"
        "    imported.\n"
        "  - No filesystem, network, subprocess, no attribute access on\n"
        "    builtins (no getattr/setattr/__class__/etc.).\n"
        "  - Return a LIST of turn_ids that is a SUBSET-OR-REORDER of the\n"
        "    input `selected_turns`. Every returned id MUST already be in\n"
        "    the input — NO fabricated ids. You may drop ids (filter) and\n"
        "    reorder them; you may not add new ones.\n"
        "  - You do not know which turns are evidence at call time — reason\n"
        "    over `scores` shape (relative ranks, gaps) and the question.\n\n"
        f"Per-question failure signals (the dropped evidence and the\n"
        f"competing turns that crowded it out):\n"
        f"{signals_block}\n\n"
        "Reply with a single ```python``` block containing only the\n"
        "function. No prose."
    )


def _build_reader_prompt(dominant_regime: str, signals_block: str) -> str:
    return (
        f"You are authoring a Python reader-prompt-transform to address the "
        f"'{dominant_regime}' regime.\n\n"
        f"Signature (REQUIRED, exact):\n  {READER_PROMPT_TRANSFORM_SIGNATURE}\n\n"
        "The transform is called once per question to edit the reader's\n"
        "prompt fragments. `prompt_parts` is a dict of named prompt pieces\n"
        "(e.g. 'context', 'instruction'). The evidence already reached the\n"
        "reader's context on the failing questions below — retrieval and\n"
        "assembly worked — but the reader still answered wrong. Your job is\n"
        "to EDIT the prompt fragment values (typically 'instruction') to\n"
        "write guidance that helps the reader reconcile the evidence that\n"
        "is already present and produce the correct answer.\n\n"
        "Constraints:\n"
        "  - Pure Python; ONLY the `math` and `string` modules may be\n"
        "    imported.\n"
        "  - No filesystem, network, subprocess, no attribute access on\n"
        "    builtins (no getattr/setattr/__class__/etc.).\n"
        "  - Return a dict with the SAME KEYS as the input `prompt_parts`.\n"
        "    You may EDIT the values; you may NOT add or remove keys.\n"
        f"  - Total added text across all fragments must be <= "
        f"{READER_PROMPT_MAX_ADDED_CHARS} characters.\n\n"
        f"Per-question reconciliation failures (evidence present in context\n"
        f"but the answer was still wrong — write instructions targeting\n"
        f"these):\n"
        f"{signals_block}\n\n"
        "Reply with a single ```python``` block containing only the\n"
        "function. No prose."
    )


def _build_author_prompt(dominant_regime: str, signals_block: str) -> str:
    return (
        f"You are authoring a Python score-transform to address the "
        f"'{dominant_regime}' retrieval regime.\n\n"
        f"Signature (REQUIRED, exact):\n  {TRANSFORM_SIGNATURE}\n\n"
        "The transform is called once per question, AFTER scoring, BEFORE\n"
        "the assembly step that walks `scores` in descending order and\n"
        "drops candidates once a token budget is exceeded. Your job is to\n"
        "REWEIGHT scores so that:\n"
        "  - evidence turns currently DROPPED AT THE BUDGET WALL survive\n"
        "    the cut on the failing questions below, AND\n"
        "  - turns that other questions rely on (the budget winners that\n"
        "    ARE evidence for some question) are NOT evicted in the\n"
        "    process.\n\n"
        "Constraints:\n"
        "  - Pure Python; ONLY the `math` module may be imported.\n"
        "  - No filesystem, network, subprocess, no attribute access on\n"
        "    builtins (no getattr/setattr/__class__/etc.).\n"
        "  - Return a dict over the SAME turn_ids as the input scores.\n"
        "  - You do not know which turns are evidence at call time — you\n"
        "    only have the score dict, the graph, and the question. Reason\n"
        "    over score shape (relative ranks, gaps, distribution).\n\n"
        f"Per-question failure signals (the cases you must move):\n"
        f"{signals_block}\n\n"
        "Reply with a single ```python``` block containing only the\n"
        "function. No prose."
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
