"""ActiveGraph-native SQL agent.

Same shape as `regimes.agent` for LongMemEval: four `@behavior`
registrations driven by the activegraph Runtime, with one configurable
pipeline (here: `prompt_transforms` instead of LME's
`score-transforms`) as the optimization seam.

Public surface:
    from regimes.targets.sql.agent import retrieve
    trace = retrieve(instance, reader=...)
    trace.predicted_sql
    trace.events
"""

from __future__ import annotations

from regimes.targets.sql.agent.agent import (
    DraftedQuery,
    RetrieveTrace,
    retrieve,
)

__all__ = ["DraftedQuery", "RetrieveTrace", "retrieve"]
