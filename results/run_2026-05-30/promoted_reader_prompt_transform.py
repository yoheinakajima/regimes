def transform(prompt_parts: dict, question: str, question_date: str) -> dict:
    import string

    extra_instruction = (
        "\n\nCRITICAL INSTRUCTIONS FOR ANSWERING:\n"
        "1. The context contains ALL the evidence needed. Do NOT say the context lacks information if relevant sessions are present.\n"
        "2. For COUNTING questions (e.g., 'how many weddings', 'how many events'): carefully scan EVERY session chunk and count ALL matching items. Sum them up and state the total.\n"
        "3. For TEMPORAL questions (e.g., 'what day of the week', 'consecutive days', 'same day'): use the session dates and the question date to reason about days of the week and relative timing. If a session has a date, compute the day of the week from it.\n"
        "4. For PRICE/DETAIL questions: look for ALL mentions of the item across ALL sessions. If one session mentions a sale price and another mentions an original price, combine them.\n"
        "5. For PREFERENCE/PLANNING questions: extract specific details (names, places, preferences) from the context and use them directly in your answer.\n"
        "6. Do NOT hedge with 'I cannot find' or 'the context does not contain' if relevant session chunks are listed in the context. Trust the evidence.\n"
        "7. If multiple sessions mention the same topic, synthesize across ALL of them.\n"
        "8. For temporal reasoning: the question date is " + str(question_date) + ". Use this to anchor relative time references like 'today', 'yesterday', 'this week'.\n"
        "9. Give a direct, specific answer. Avoid vague or non-committal responses.\n"
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
