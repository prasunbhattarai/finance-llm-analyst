from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def vectorstore(docs, persist_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create a Chroma-based vector store with embeddings and a reranking retriever.

    Steps:
        1. Initialize Hugging Face sentence embeddings for document encoding.
        2. Build (or reuse) a Chroma vector database in the given directory.
        3. Add documents if the collection is empty, otherwise reuse existing ones.
        4. Attach a Hugging Face cross-encoder reranker for contextual compression.
        5. Return a retriever that retrieves top-k documents with reranking.

    Args:
        docs (List[Document]): Documents to be added to the vector store if empty.
        persist_dir (str): Directory path where the Chroma database is persisted.
        model_name (str, optional): Name of the Hugging Face embeddings model.
                                    Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        ContextualCompressionRetriever: A retriever that performs semantic search with reranking.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    collection_count = vectordb._collection.count()
    if collection_count == 0:
        vectordb.add_documents(docs)
        print(f"Added {len(docs)} documents to Chroma.")
    else:
        print(f"Vector store already has {collection_count} documents. Skipping addition.")

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    reranker = CrossEncoderReranker(model=cross_encoder)

    retriever = ContextualCompressionRetriever(
        base_retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        base_compressor=reranker
    )

    return retriever
