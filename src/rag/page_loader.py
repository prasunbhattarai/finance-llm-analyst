from langchain_community.document_loaders import WebBaseLoader

def load_websites(urls: list[str]):
    docs = []
    for url in urls:
        loader = WebBaseLoader(urls)
        loaded_docs= loader.load()

        for doc in loaded_docs:
            doc.metadata["source_url"]= url
        docs.extend(loaded_docs)
    return docs