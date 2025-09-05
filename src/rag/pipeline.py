from .multi_query import multi_query
from .page_loader import load_websites
from .vectorstore import vectorstore
from .base import load_finetuned_llm
from langchain_core.prompts import PromptTemplate



def pipiline(cfg,question,urls):
    
    docs = load_websites(urls)

    retriever = vectorstore(docs)

    query_chain = multi_query(cfg)

    queries = query_chain.invoke({"question": question})

    retrieved_docs = []

    for q in queries:
        retrieved_docs.extend(retriever.get_relevant_documents(q))

    unique_docs = {doc.page_content: doc for doc in retrieved_docs}.values()
    context = "\n\n".join(doc.page_content for doc in unique_docs)

    llm = load_finetuned_llm(cfg)

    prompt = PromptTemplate(
        input_variables=["context","question"],
        template="""
        You are FIN-LLM, a world-class financial analyst with expertise in corporate finance, 
        investment research, and data-driven decision making. 
        You provide precise, professional, and trustworthy answers.
        
        Instructions:
        - Use only the information from the context below.  
        - If the answer is not present or cannot be derived, respond with: 
          "The available data is insufficient to answer this question."  
        - Always explain reasoning step by step before giving the final answer.  
        - Structure your response clearly with headings, bullet points, or tables when helpful.  
        - Keep the tone formal and concise, like an equity research report.  
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer (with reasoning and conclusion):
        """
    )
    chain = prompt | llm
    answer = chain.invoke({"context":context, "question":question})
    
    return answer
