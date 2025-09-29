import re

def is_finance_question(question: str, llm) -> bool:
    """
    Determine whether a user question is finance-related using an LLM classifier.

    The function prompts the language model to strictly classify a question as either
    "finance" or "not". It then extracts and normalizes the LLM's response to return
    a boolean.

    Args:
        question (str): The user-provided question.
        llm: A language model interface.

    Returns:
        bool: True if the question is classified as "finance", False otherwise.
    """

    check_prompt = f"""
    Classify the following user question strictly as 'finance' or 'not finance'.
    Only return one word: either finance or not.
    Question: "{question}"
    Answer:
    """

    result = llm.invoke(check_prompt)

    if isinstance(result, dict):
        result = result.get("text") or str(result)

    result = result.strip().lower()

    if "answer:" in result:
        result = result.split("answer:")[-1].strip()

    first_word = result.split()[0] if result else ""
    first_word = re.sub(r'[^a-z]', '', first_word)

    return first_word == "finance"
