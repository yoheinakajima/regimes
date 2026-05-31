def transform(prompt_parts: dict, question: str, question_date: str) -> dict:
    import string

    additional_instruction = (
        "\n\nCRITICAL INSTRUCTIONS FOR ANSWERING:\n"
        "1. The context provided ALREADY CONTAINS the answer. Search every passage carefully before saying information is missing.\n"
        "2. For knowledge-update questions (conflicting info across sessions): ALWAYS use the MOST RECENT session's information as the final answer. Do not report a conflict — just state the latest value.\n"
        "3. For temporal-reasoning questions: Look for dates, ages, years mentioned anywhere in the context. Compute the answer using the question date if needed. If a birth year and an event year are present, subtract to find age.\n"
        "4. For multi-session questions: Combine information across ALL sessions. If a calculation is needed (e.g. discount percentage), perform it: percentage = ((original - paid) / original) * 100.\n"
        "5. For single-session-preference questions: The answer IS in the context. Re-read every fragment. Do not say the context lacks the answer.\n"
        "6. NEVER say 'the context does not contain' or 'I cannot find' if evidence passages are present — they contain the answer.\n"
        "7. Give a direct, concise answer. Do not hedge or list conflicts unless explicitly asked.\n"
        "8. If two sessions give different values for the same fact, the LATER session's value is correct — use it.\n"
    )

    result = {}
    for key, value in prompt_parts.items():
        if key == 'instruction':
            result[key] = value + additional_instruction
        else:
            result[key] = value

    return result