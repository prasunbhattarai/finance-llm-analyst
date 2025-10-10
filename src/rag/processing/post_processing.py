def post_processing(answer: str) -> str:
    """
    Clean and trim the model's output to a maximum of three sentences.

    Steps:
        1. Handle cases where the answer is returned as a dict.
        2. Remove any leading "Answer:" prefix.
        3. Split the text into sentences (by '.').
        4. Keep at most the first three sentences.
        5. Return the cleaned, trimmed string.

    Args:
        answer : Raw model output.

    Returns:
        str: Processed answer limited to three sentences.
    """
    if isinstance(answer, dict):
        answer = answer.get("text") or answer.get("output") or str(answer)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]

    sentences = [s.strip() for s in answer.split('.') if s.strip()]

    trimmed = '. '.join(sentences[:5]) + ('.' if sentences else '')

    return trimmed
