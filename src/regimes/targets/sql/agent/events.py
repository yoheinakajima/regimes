"""SQL agent event vocabulary.

The chain (one question):
    question.asked
      -> sql_agent.encode_schema      -> schema.encoded
      -> sql_agent.retrieve_relevant_columns -> columns.scored
      -> sql_agent.prompt_pipeline    -> prompt.assembled
      -> sql_agent.draft_query        -> query.drafted
"""

from __future__ import annotations

QUESTION_ASKED = "question.asked"
SCHEMA_ENCODED = "schema.encoded"
COLUMNS_SCORED = "columns.scored"
PROMPT_ASSEMBLED = "prompt.assembled"
QUERY_DRAFTED = "query.drafted"
