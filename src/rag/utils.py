def post_processing(answer):
    if isinstance(answer, dict):
        answer = answer.get("text") or answer.get("output") or str(answer)

    # Handle special cases
    if "I can only answer finance" in answer.lower():
        return "I can only answer finance related questions"

    # Clean out system-style echoes
    if "You are an expert financial assistant" in answer:
        answer = answer.split("Answer:")[-1].strip() if "Answer:" in answer else answer

    return answer.strip()
