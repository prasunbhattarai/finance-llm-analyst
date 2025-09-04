from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from .base import load_finetuned_llm

class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text):
        lines = text.strip().split("\n")
        return list(filter(None,lines))
    
def multi_query(cfg):
    llm= load_finetuned_llm(cfg)
    output_parser= LineListOutputParser()

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",    
    )
    llm_chain= query_prompt | llm | output_parser
    return llm_chain
