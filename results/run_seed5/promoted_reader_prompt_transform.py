def transform(prompt_parts: dict, question: str, question_date: str) -> dict:
    import string

    extra_instruction = (
        "\n\nCRITICAL INSTRUCTIONS FOR ANSWERING:\n"
        "1. The context provided contains the evidence needed to answer the question. "
        "Read ALL context fragments carefully before concluding that information is absent.\n"
        "2. Do NOT say 'the context does not contain information' if ANY relevant detail "
        "appears anywhere in the context — even partially, indirectly, or across multiple fragments.\n"
        "3. For preference/recommendation questions: look for any mention of shows, movies, books, "
        "restaurants, or activities the user expressed interest in or was recommended.\n"
        "4. For multi-session questions: combine information across ALL context fragments. "
        "Each fragment may hold a piece of the answer.\n"
        "5. For temporal-reasoning questions: carefully compute dates and durations using "
        "the question date and any dates mentioned in the context.\n"
        "6. For personal-fact questions (age, name, relationships): extract any stated or "
        "implied personal details from the context, even if mentioned briefly.\n"
        "7. If the evidence is present but ambiguous, make your best inference and state it "
        "clearly rather than claiming no information exists.\n"
        "8. Answer based on what IS in the context, not on what you expect to find.\n"
    )

    result = {}
    for key, value in prompt_parts.items():
        if key == 'instruction':
            result[key] = value + extra_instruction
        else:
            result[key] = value

    return result