def transform(prompt_parts: dict, question: str, question_date: str) -> dict:
    import string

    extra_instruction = (
        "\n\nCRITICAL INSTRUCTIONS FOR ANSWERING:\n"
        "1. READ ALL EVIDENCE CAREFULLY: The context contains retrieved evidence chunks. "
        "Scan every chunk thoroughly before answering.\n"
        "2. EXACT ANSWERS ONLY: Give a concise, direct answer. Do NOT pad with background "
        "or unrelated items. Only include what the evidence explicitly states.\n"
        "3. TEMPORAL REASONING: When the question involves dates, times, or sequences, "
        "carefully compute differences, check consecutive days, and verify booking/event "
        "timelines using the exact dates in the evidence. Do not guess or approximate.\n"
        "4. MULTI-SESSION: If evidence spans multiple sessions, reconcile ALL relevant "
        "chunks. The correct answer may require combining information from different sessions.\n"
        "5. SINGLE ITEM QUESTIONS: If the question asks for a specific item (e.g., a gift, "
        "a purchase), return ONLY that item as stated in the evidence — do not list extras.\n"
        "6. IF EVIDENCE IS PRESENT: Trust the retrieved evidence over general knowledge. "
        "The answer IS in the context — find it and state it directly.\n"
        "7. CONSECUTIVE DAYS CHECK: For 'consecutive days' questions, list all event dates "
        "and check if any two differ by exactly 1 day.\n"
        "8. DATE ARITHMETIC: For 'how many months/weeks/days in advance' questions, "
        "subtract the booking date from the event date precisely.\n"
        "9. DO NOT SAY 'no mention' if evidence chunks are present — re-read them.\n"
        "10. FINAL ANSWER: State the answer in one or two sentences maximum.\n"
    )

    result = dict(prompt_parts)

    if 'instruction' in result:
        result['instruction'] = result['instruction'] + extra_instruction
    elif 'context' in result:
        result['context'] = result['context'] + extra_instruction
    else:
        # Add to the first available key
        keys = list(result.keys())
        if keys:
            result[keys[0]] = result[keys[0]] + extra_instruction

    return result