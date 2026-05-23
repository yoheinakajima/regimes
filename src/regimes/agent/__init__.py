"""Runtime-native ActiveGraph retrieval agent.

The four behaviors and the ingest path use the published `activegraph`
package directly: real Graph, real @behavior decorations, real Runtime,
real graph.emit. There is no lookalike engine here.

Public surface:
    from regimes.agent import retrieve
    trace = retrieve(instance, token_budget=2500)
    trace.context.text       # the assembled context
    trace.events             # full runtime event log for this question
"""

from __future__ import annotations

from regimes.agent.agent import (
    AssembledContext,
    RetrieveTrace,
    DEFAULT_MAX_DOC_FREQ_FRACTION,
    DEFAULT_MIN_SESSION_COOCCURRENCE,
    DEFAULT_MIN_TOKEN_LENGTH,
    DEFAULT_TOKEN_BUDGET,
    ingest,
    retrieve,
)

__all__ = [
    "AssembledContext",
    "RetrieveTrace",
    "DEFAULT_MAX_DOC_FREQ_FRACTION",
    "DEFAULT_MIN_SESSION_COOCCURRENCE",
    "DEFAULT_MIN_TOKEN_LENGTH",
    "DEFAULT_TOKEN_BUDGET",
    "ingest",
    "retrieve",
]
