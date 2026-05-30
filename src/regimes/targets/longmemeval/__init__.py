"""LongMemEval target — the existing eval loop, now behind the
generalized `regimes.target.Target` interface.

Public surface:

    from regimes.targets.longmemeval import LongMemEvalTarget, build_target

`build_target` is the convenience constructor the loop's runner uses to
turn `(eval_backend, author)` into a full Target.
"""

from __future__ import annotations

from regimes.targets.longmemeval.action_space import (
    LONGMEMEVAL_IMPORT_WHITELIST,
    LONGMEMEVAL_PER_TYPE_FLOORS,
    LONGMEMEVAL_SIGNATURE_PARAMS,
    LongMemEvalActionSpace,
)
from regimes.targets.longmemeval.outcome_summary import outcome_summary
from regimes.targets.longmemeval.target import LongMemEvalTarget, build_target
from regimes.targets.longmemeval.taxonomy import LongMemEvalTaxonomy
from regimes.targets.longmemeval.transform_types import (
    ALL_TRANSFORM_TYPES,
    ASSEMBLY_TRANSFORM,
    READER_PROMPT_TRANSFORM,
    REGIME_TO_TYPES,
    SCORE_TRANSFORM,
    TransformType,
)

__all__ = [
    "ALL_TRANSFORM_TYPES",
    "ASSEMBLY_TRANSFORM",
    "LONGMEMEVAL_IMPORT_WHITELIST",
    "LONGMEMEVAL_PER_TYPE_FLOORS",
    "LONGMEMEVAL_SIGNATURE_PARAMS",
    "LongMemEvalActionSpace",
    "LongMemEvalTarget",
    "LongMemEvalTaxonomy",
    "READER_PROMPT_TRANSFORM",
    "REGIME_TO_TYPES",
    "SCORE_TRANSFORM",
    "TransformType",
    "build_target",
    "outcome_summary",
]
