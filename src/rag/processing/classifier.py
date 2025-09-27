import re
def is_finance_question(question, llm):
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
   # extract only what comes after "answer:"
   if "answer:" in result:
       result = result.split("answer:")[-1].strip()
   # only take the first word (ignore explanations)
   first_word = result.split()[0] if result else ""
   first_word = re.sub(r'[^a-z]', '', first_word)
   return first_word == "finance"
