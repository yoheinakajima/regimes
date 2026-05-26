"""Eval backends: the loop's bridge to a scoring function.

`real`  — wraps the LME harness: runs the agent + a reader, writes
          hypotheses.jsonl in LME's format, shells out to LME's frozen
          judge, parses scores.json + per-question judgments, returns
          Outcome records the loop's diagnose step consumes.

The Outcome / EvalResult shape (see `types.py`) is the single contract
between any eval backend and the loop. MockEval (built later in
milestone 3) will produce the same shape.
"""

from __future__ import annotations

from regimes.eval.types import (
    Outcome,
    EvalResult,
    Reader,
    Judge,
)
from regimes.eval.real import (
    RealEval,
    FakeReader,
    FakeJudge,
    AnthropicReader,
    LMEJudge,
    build_real_reader,
)

__all__ = [
    "Outcome",
    "EvalResult",
    "Reader",
    "Judge",
    "RealEval",
    "FakeReader",
    "FakeJudge",
    "AnthropicReader",
    "LMEJudge",
    "build_real_reader",
]
