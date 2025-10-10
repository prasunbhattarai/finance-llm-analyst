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
    You are a financial domain classifier.
    Classify the following question as 'finance' if it relates to money, banking,
    investing, markets, accounting, business, economics, taxes, or financial planning.
    Otherwise classify it as 'not finance'.

    Only respond with one word: 'finance' or 'not'.
    Question: "{question}"
    Answer:
    """

    result = llm.invoke(check_prompt)

    match = re.search(r'Answer\s*[:\-]?\s*([A-Za-z]+)', result, re.IGNORECASE)
    first_word = match.group(1).lower() 

    # print("Extracted:", first_word)
    # print("Raw result:", result)

    return first_word == "finance"
