from .multi_query import multi_query 
from .page_loader import load_websites, split_docs
from .csv_loader import csv_to_docs
from .vectorstore import vectorstore
from .base import loaded_model
from langchain_core.prompts import PromptTemplate
import transformers
import re


transformers.logging.set_verbosity_error()    


class Pipeline:
    def __init__(self, cfg, llm):
        self.cfg = cfg
        self.llm = llm

        docs = split_docs(load_websites())
        self.web_retriever = vectorstore(docs, persist_dir="./chroma_db/webpages")

        stocks_docs = csv_to_docs(cfg["Stocks"]["stocks_dataset"])
        self.stock_retriever = vectorstore(stocks_docs, persist_dir="./chroma_db/stocks")

        self.query_chain = multi_query(cfg)
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert financial assistant. "
                "Use ONLY the provided context to answer the user question in 3–4 sentences. \n\n"
                "Your ENTIRE answer must be no longer than 3 sentences. "
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Answer:"
            ),
        )

        self.chain = self.prompt | self.llm
    def is_finance_question(self, question):
       check_prompt = f"""
       Classify the following user question strictly as 'finance' or 'not finance'.
       Only return one word: either finance or not.

       Question: "{question}"
       Answer:
       """
       result = self.llm.invoke(check_prompt)

       if isinstance(result, dict):
           result = result.get("text") or str(result)

       result = result.strip().lower()

       # extract only what comes after "answer:"
       if "answer:" in result:
           result = result.split("answer:")[-1].strip()

       # only take the first word (ignore explanations)
       first_word = result.split()[0] if result else ""
       first_word = re.sub(r'[^a-z]', '', first_word)
       print(f"[DEBUG] Raw: {result} | Parsed: {first_word}")

       return first_word == "finance"

    
    def post_processing(self, answer):
        if isinstance(answer, dict):
            answer = answer.get("text") or answer.get("output") or str(answer)

        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1]

        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        trimmed = '. '.join(sentences[:3]) + ('.' if sentences else '')

        return trimmed

    
    def ask(self, question):
        queries = self.query_chain.invoke({"question": question})
        retrieved_docs = []

        for q in queries:
            if "stock" in question.lower() or "price" in question.lower():
                retrieved_docs.extend(self.stock_retriever.invoke(q))
            else:
                retrieved_docs.extend(self.web_retriever.invoke(q))

        # remove duplicates
        unique_docs = list({doc.page_content: doc for doc in retrieved_docs}.values())

        context = "\n\n".join(doc.page_content for doc in unique_docs) if unique_docs else "No relevant documents found."

        raw_answer = self.chain.invoke({"context": context, "question": question})
        clean_answer = self.post_processing(raw_answer)

        return clean_answer


def main(cfg):
    llm = loaded_model(cfg)
    pipeline = Pipeline(cfg, llm)
    print("Finance Analyst ready. Type 'exit' to quit")

    while True:
        question = input("Enter your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        # classify first
        if not pipeline.is_finance_question(question):
            print("I can only answer finance related questions.\n")
            continue  
        else:
            answer = pipeline.ask(question)
        print("\n=== Final Answer ===\n")
        print(answer + "\n")



