from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from .base import load_finetuned_llm

class DecompositionOutputParser(BaseOutputParser[List[str]]):
    
    def parse(self, text):
        lines = text.strip().split("\n")
        return list(filter(None,lines))
    

def decompose_query(cfg):
    llm = load_finetuned_llm(cfg)
    output_parser = DecompositionOutputParser()

    decompostion_prompt= PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant specialized in query decomposition.
        Break down the following complex user question into a step-by-step list of 
        smaller, more specific sub-questions that can be individually answered.
        
        Complex question: {question}
        Sub-questions:""",
    )
    llm_chain = decompostion_prompt | llm | output_parser
    return llm_chain

