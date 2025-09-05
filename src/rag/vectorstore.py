from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from sentence_transformers import CrossEncoder


def vectorstore(docs, persist_dir="./chroma_db", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name= model_name)

    vectordb = Chroma.from_documents(
        embedding= embeddings,
        persist_directory= persist_dir
    )


    collection_count = vectordb._collection.count()
    if collection_count ==0:
        vectordb.add_documents(docs)
        vectordb.persist()
        print(f"Added {len(docs)} documents to Chroma.")
    else:
        print(f"Vector store already has {collection_count} documents. Skipping addition.")

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   
    reranker= CrossEncoderReranker(model=cross_encoder)

    retriever = ContextualCompressionRetriever(
        base_retriever= vectordb.as_retriever(search_kwargs={"k":10}),
        base_compressor= reranker
    )
    
    return retriever