from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_html(raw_html):
    
    soup = BeautifulSoup(raw_html,"html.parser")

    for tag in soup(["nav", "footer", "header", "script", "style", "noscript", "iframe", "form", "button"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return " ".join(text.split())

urls = [
    "https://www.investopedia.com/terms/c/compoundinterest.asp",
    "https://www.investopedia.com/terms/i/inflation.asp",
    "https://www.investopedia.com/terms/e/etf.asp",
    "https://www.federalreserve.gov/monetarypolicy/openmarket.htm",
    "https://www.kiplinger.com/investing/diy-investors-dont-make-these-mistakes",
    "https://www.investopedia.com/the-federal-reserve-meeting-starts-today-what-you-need-to-know-11810701",
    "https://www.investopedia.com/market-faces-usd1-5-trillion-downside-if-trump-fires-fed-chair-powell-study-warns-11809083",
    "https://www.investopedia.com/investing-in-cryptocurrency-5215269",
    "https://www.investopedia.com/articles/personal-finance/081514/what-do-credit-score-ranges-mean.asp",
    "https://www.investopedia.com/the-hidden-costs-of-ignoring-your-taxes-what-you-should-be-aware-of-11799289"
]

def load_websites():
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