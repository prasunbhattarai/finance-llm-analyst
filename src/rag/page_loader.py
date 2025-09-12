from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_html(raw_html):
    
    soup = BeautifulSoup(raw_html,"html.parser")

    for tag in soup(["nav", "footer", "header", "script", "style", "noscript", "iframe", "form", "button"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return " ".join(text.split())

def load_websites(urls: list[str]):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        loaded_docs= loader.load()

        for doc in loaded_docs:
            doc.page_content = clean_html(doc.page_content)
            doc.metadata["source_url"]= url
        docs.extend(loaded_docs)
    return docs

def split_docs(docs, chunk_size=1000, chunk_overlap=100):
    """Split documents into chunks for better embeddings & retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)