from .multi_query import multi_query
from .page_loader import load_websites, split_docs
from .csv_loader import csv_to_docs
from .vectorstore import vectorstore
from .base import loaded_model
from .utils import post_processing
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
import transformers


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
        input_variables=["context","question"],
        template = """
        You are an expert financial assistant. Use ONLY the provided context  to answer the user question. Each answer can only be of 3-4 sentences. 
        If the answer is not contained in the snippets, say you don't know and give a short explanation of what additional 
        information would be needed.\n\n"

        Question:
        {question}

        Context:
        {context}

        Answer:

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
        # unique_docs = unique_docs[:3]   
        if unique_docs:
            context = "\n\n".join(doc.page_content for doc in unique_docs)
        else:
            context = "No relevant documents found."

        answer = self.chain.invoke({"context":context, "question":question})
        
        answer = post_processing(answer)

        return answer

def main(cfg):
    llm = loaded_model(cfg)
    pipeline= Pipeline(cfg, llm)
    print("Finance Analyts ready. Type 'exit' to quit")
    while True:
        question = input("Enter your question: ").strip()

        if question.lower() in ("exit","quit"):
            break
        answer = pipeline.ask(question)
        print("\n===Final Answer===\n")
        print(f"{answer}\n") 