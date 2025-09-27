def post_processing( answer):
    if isinstance(answer, dict):
        answer = answer.get("text") or answer.get("output") or str(answer)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    trimmed = '. '.join(sentences[:3]) + ('.' if sentences else '')
    return trimmed
