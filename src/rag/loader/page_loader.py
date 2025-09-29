from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_html(raw_html):
    """
    Clean raw HTML content by removing unnecessary elements and extracting plain text.

    The following tags are removed entirely:
    nav, footer, header, script, style, noscript, iframe, form, button.

    Args:
        raw_html (str): Raw HTML string to process.

    Returns:
        str: Cleaned and whitespace-normalized plain text.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

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
    """
    Load and clean text documents from predefined finance-related URLs.

    - Uses `WebBaseLoader` to fetch page content.
    - Cleans raw HTML with `clean_html`.
    - Stores the original URL in each document's metadata under `source_url`.

    Returns:
        list[Document]: List of cleaned LangChain Document objects.
    """
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.page_content = clean_html(doc.page_content)
            doc.metadata["source_url"] = url
        docs.extend(loaded_docs)
    return docs


def split_docs(docs, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller overlapping chunks for embedding and retrieval.

    Args:
        docs (list[Document]): List of LangChain Document objects.
        chunk_size (int, optional): Maximum size of each text chunk. Default is 1000.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Default is 100.

    Returns:
        list[Document]: List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)
