from .multi_query import multi_query
from .page_loader import load_websites, split_docs
from .csv_loader import csv_to_docs
from .vectorstore import vectorstore
from .base import loaded_model
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
import yaml
import os

def load_config(filename: str):
    """
    Load a YAML config file from the project's configs directory.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, "configs", filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as cfgs:
        return yaml.safe_load(cfgs)
    
class Pipeline:
    def __init__(self, cfg, urls, llm):
        self.cfg = cfg
        self.urls = urls
        self.llm = llm

        docs = split_docs(load_websites(urls))
        self.web_retriever = vectorstore(docs, persist_dir="./chroma_db/webpages")

        stocks_docs = csv_to_docs(cfg["Stocks"]["stocks_dataset"])
        self.stock_retriever = vectorstore(stocks_docs, persist_dir="./chroma_db/stocks")


        self.query_chain = multi_query(cfg)
        self.prompt = PromptTemplate(
        input_variables=["context","question"],
        template = """
        Answer the following question using the context below. 
        Use step-by-step reasoning and give a final conclusion.
        If context is empty, answer based on your own knowledge.

        Question:
        {question}

        Context (may be empty):
        {context}

        Answer (with reasoning and conclusion):
        """
        )

        self.chain = self.prompt | self.llm
        
    
    def ask(self,question):
        queries = self.query_chain.invoke({"question": question})
        retrieved_docs = []

        for q in queries:
            if "stock" in question.lower() or "price" in question.lower():
                retrieved_docs.extend(self.stock_retriever.invoke(q))
            else:
                retrieved_docs.extend(self.web_retriever.invoke(q))            


        unique_docs = list({doc.page_content: doc for doc in retrieved_docs}.values())
        # limited_docs = unique_docs[:3]   # take only top 3 (adjust if GPU allows more)
        if unique_docs:
            context = "\n\n".join(doc.page_content for doc in unique_docs)
        else:
            context = "No relevant documents found."

        answer = self.chain.invoke({"context":context, "question":question})
        
        return answer



if __name__ == "__main__":
    cfg = load_config("qwen.yaml")


    urls = [
    "https://www.investopedia.com/terms/c/compoundinterest.asp",
    "https://www.investopedia.com/terms/i/inflation.asp",
    "https://www.investopedia.com/terms/e/etf.asp",
    "https://www.federalreserve.gov/monetarypolicy/openmarket.htm"
    ]
    llm = loaded_model(cfg)
    pipeline= Pipeline(cfg, urls, llm)
    print("Finance Analyts ready. Type 'exit' to quit")
    while True:
        question = input("Enter your question: ").strip()

        if question.lower() in ("exit","quit"):
            break
        answer = pipeline.ask(question)
        print("\n===Final Answer===\n")
        print(answer) 