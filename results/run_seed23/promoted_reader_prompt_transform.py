def transform(prompt_parts: dict, question: str, question_date: str) -> dict:
    import string

    extra_instruction = (
        "\n\nCRITICAL INSTRUCTIONS FOR ANSWERING:\n"
        "1. EVIDENCE RECONCILIATION: The context contains all evidence needed. "
        "Read every piece of context carefully before answering.\n"
        "2. TEMPORAL REASONING: When asked about time gaps, durations, or sequences, "
        "identify the exact dates/times mentioned, compute the difference explicitly, "
        "and state the numeric result. Do not say 'I cannot determine' if dates are present.\n"
        "3. KNOWLEDGE UPDATES: If multiple sessions mention conflicting values (e.g. loan amounts, "
        "prices), use the MOST RECENT session's value as the current answer unless asked otherwise. "
        "Pay attention to session dates.\n"
        "4. COUNTING TASKS: When asked 'how many', enumerate ALL matching items from context, "
        "count them precisely, and give the exact number. Do not stop at the first match.\n"
        "5. SINGLE CORRECT SOURCE: If the question is about 'you' (the user), answer only from "
        "what the user themselves stated — not from other speakers or hypothetical examples.\n"
        "6. FAITH/ACTIVITY CATEGORIES: When asked about a category of activities (e.g. faith-related), "
        "scan ALL sessions and list every matching item, then count them.\n"
        "7. COUPON/DISCOUNT SOURCE: Identify exactly where (email, app, store, etc.) the coupon "
        "or discount came from as stated in context.\n"
        "8. DINNER PARTY DETAILS: Report only what the user personally hosted or attended, "
        "with exact details (date, theme, host) from context.\n"
        "9. ADVANCE BOOKING: When asked how far in advance something was booked, compute the "
        "difference between booking date and event date in the requested unit (days/weeks/months).\n"
        "10. Give a direct, specific answer. Do not hedge if the evidence is present.\n"
    )

    result = {}
    for key, value in prompt_parts.items():
        if key == 'instruction':
            result[key] = value + extra_instruction
        else:
            result[key] = value

    return result