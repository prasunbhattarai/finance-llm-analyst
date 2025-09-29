from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from src.rag.model import loaded_model

class LineListOutputParser(BaseOutputParser[List[str]]):
    """
    Custom output parser that converts raw LLM output into a list of non-empty lines.

    Example:
        Input: "Question 1\nQuestion 2\n\nQuestion 3"
        Output: ["Question 1", "Question 2", "Question 3"]
    """

    def parse(self, text):
        lines = text.strip().split("\n")
        return list(filter(None, lines))


def multi_query(cfg):
    """
    Build a LangChain pipeline that generates multiple reformulations of a user question.

    This function:
        1. Loads the language model from configuration.
        2. Defines a prompt instructing the LLM to create five alternative phrasings
           of the original question.
        3. Parses the output into a clean list of queries.

    Args:
        cfg (dict): Configuration dictionary for loading the model.

    Returns:
        RunnableSequence: A LangChain chain that, when invoked with {"question": str},
                          returns a list of reformulated queries.
    """
    llm = loaded_model(cfg)
    output_parser = LineListOutputParser()

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    llm_chain = query_prompt | llm | output_parser
    return llm_chain
