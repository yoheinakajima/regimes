def transform(prompt_parts: dict, question: str, question_date: str) -> dict:
    import string

    extra_instruction = (
        "\n\n=== CRITICAL RECONCILIATION GUIDANCE ===\n"
        "The context already contains the evidence needed to answer. Follow these steps:\n\n"
        "1. SESSION DATES: Each session chunk has a header like 'Session answer_XXXX_N' with a date/time stamp. "
        "Use these dates to establish the ABSOLUTE timeline of events. Do NOT say you cannot determine order if session dates are present.\n\n"
        "2. RELATIVE TIME WORDS: Words like 'yesterday', 'last week', 'tomorrow' are RELATIVE to the session date. "
        "Resolve them: if a session is dated 2022/03/10 and mentions 'yesterday', that event was 2022/03/09.\n\n"
        "3. COUNTING ACROSS SESSIONS: When asked 'how many times' or 'how many events', scan ALL session chunks "
        "and count every distinct occurrence. Do not stop at the first mention.\n\n"
        "4. CONSECUTIVE DAYS: To check if two events were on consecutive days, resolve each event to its absolute date "
        "using the session date + any relative references, then compare.\n\n"
        "5. GIFT / DETAIL QUESTIONS: If asked about a specific detail (e.g. a gift amount), look across ALL sessions "
        "for every mention and report the one that matches the question's context.\n\n"
        "6. DO NOT say 'the context does not contain' if session chunks are present — re-read them carefully.\n\n"
        "7. Provide a direct, specific answer. If multiple sessions mention the topic, synthesize them.\n"
        "=== END GUIDANCE ===\n"
    )

    result = {}
    for key, value in prompt_parts.items():
        if key == 'instruction':
            result[key] = value + extra_instruction
        else:
            result[key] = value

    return result