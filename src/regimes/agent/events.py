"""Agent event vocabulary.

The runtime-native retrieval agent emits a fixed sequence of custom event
types through the real activegraph runtime. These are constants, not
strings sprinkled across files, so the loop's regime detectors can match
on them safely.

Sequence per question:
    question.asked
      -> behavior_score fires
    turns.scored
      -> behavior_transform fires
    turns.transformed
      -> behavior_expand fires
    turns.expanded
      -> behavior_assemble fires
    context.assembled
      [terminal — the retrieval result lives in this event's payload]
"""

from __future__ import annotations

QUESTION_ASKED = "question.asked"
TURNS_SCORED = "turns.scored"
TURNS_TRANSFORMED = "turns.transformed"
TURNS_EXPANDED = "turns.expanded"
CONTEXT_ASSEMBLED = "context.assembled"

ALL = (
    QUESTION_ASKED,
    TURNS_SCORED,
    TURNS_TRANSFORMED,
    TURNS_EXPANDED,
    CONTEXT_ASSEMBLED,
)
