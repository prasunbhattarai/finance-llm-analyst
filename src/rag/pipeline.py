from src.rag.model import loaded_model
from src.rag.retriever import multi_query
from src.rag.loader import load_websites, split_docs, csv_to_docs
from src.rag.retriever import vectorstore
from src.rag.model.base import loaded_model
from src.rag.processing import post_processing, is_finance_question
from langchain_core.prompts import PromptTemplate
import transformers

transformers.logging.set_verbosity_error()


class Pipeline:
    """
    Finance RAG pipeline for retrieving relevant documents and generating answers.

    This class integrates:
    - Web and stock document retrievers
    - Multi-query expansion for better retrieval
    - A prompt template for consistent, concise answers
    - A language model chain for final response generation
    """

    def __init__(self, cfg: dict, llm):
        """
        Initialize the pipeline with configuration and language model.

        Args:
            cfg (dict): Configuration dictionary containing dataset paths and settings.
            llm: Loaded language model instance for inference.
        """
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

    def ask(self, question: str) -> str:
        """
        Answer a finance-related user question using retrieval + generation.

        Steps:
            1. Expand the question using multi-query retrieval.
            2. Retrieve documents from stock or web sources.
            3. Remove duplicate documents.
            4. Generate a concise answer with the LLM.
            5. Post-process the output for clarity.

        Args:
            question (str): User input question.

        Returns:
            str: Final cleaned answer from the pipeline.
        """
        queries = self.query_chain.invoke({"question": question})
        retrieved_docs = []

        for q in queries:
            if "stock" in question.lower() or "price" in question.lower():
                retrieved_docs.extend(self.stock_retriever.invoke(q))
            else:
                retrieved_docs.extend(self.web_retriever.invoke(q))

        # Remove duplicates
        unique_docs = list({doc.page_content: doc for doc in retrieved_docs}.values())
        context = (
            "\n\n".join(doc.page_content for doc in unique_docs)
            if unique_docs
            else "No relevant documents found."
        )

        raw_answer = self.chain.invoke({"context": context, "question": question})
        clean_answer = post_processing(raw_answer)
        return clean_answer


def main(cfg: dict):
    """
    Run the finance analyst assistant interactively.

    Args:
        cfg (dict): Configuration dictionary for model, retriever, and datasets.
    """
    llm = loaded_model(cfg)
    pipeline = Pipeline(cfg, llm)
    print("\nFinance Analyst ready. Type 'exit' to quit\n")

    while True:
        question = input("Enter your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break

        if not is_finance_question(question, llm):
            print("I can only answer finance related questions.\n")
            continue
        else:
            answer = pipeline.ask(question)

        print("\n=== Final Answer ===\n")
        print(answer + "\n")
