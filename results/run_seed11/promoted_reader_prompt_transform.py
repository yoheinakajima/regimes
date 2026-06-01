def transform(prompt_parts: dict, question: str, question_date: str) -> dict:
    import string

    additional_instruction = (
        "\n\nCRITICAL INSTRUCTIONS FOR ANSWERING:\n"
        "1. CAREFULLY READ ALL context fragments provided. The answer IS present in the context - do not say it is missing.\n"
        "2. For MULTI-SESSION questions: Synthesize information across ALL sessions. Count or aggregate items from every session, not just one.\n"
        "3. For TEMPORAL-REASONING questions: Use dates and ages mentioned across sessions to compute the answer mathematically. If the user's birth year or age at a known date is present, calculate accordingly.\n"
        "4. For PREFERENCE/SINGLE-SESSION questions: If the question asks about a specific topic (e.g., hotels in Miami), look carefully through ALL context fragments - the relevant information may be present even if other topics dominate.\n"
        "5. NEVER say 'the context does not contain' or 'I cannot find' if evidence fragments are present - re-read carefully.\n"
        "6. When counting events, items, or amounts across multiple sessions, enumerate EACH session's contribution separately, then sum them.\n"
        "7. For coupon/discount questions: identify the exact item and source (e.g., email, app) from the context.\n"
        "8. Prioritize the MOST RECENT session's data when sessions conflict, unless the question asks for totals or history.\n"
        "9. Give a direct, specific answer based on what the context says - do not hedge if evidence is present.\n"
    )

    result = dict(prompt_parts)

    if 'instruction' in result:
        result['instruction'] = result['instruction'] + additional_instruction
    elif 'context' in result:
        result['context'] = result['context'] + additional_instruction
    else:
        # Add to the first available key
        keys = list(result.keys())
        if keys:
            result[keys[0]] = result[keys[0]] + additional_instruction

    return result