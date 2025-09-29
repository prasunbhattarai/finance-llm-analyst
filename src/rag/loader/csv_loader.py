import pandas as pd
from langchain.schema import Document

def csv_to_docs(csv_path):
    """
    Convert a CSV file of financial data into LangChain Document objects.

    Each row is formatted into a readable text summary with metadata attached.

    Args:
        csv_path (str): Path to the CSV file containing company data.
                        Expected columns: ['Company', 'Ticker', 'Price', 'P/E Ratio', 'Market Cap (B)'].

    Returns:
        list[Document]: List of LangChain Document objects with text content and metadata.
    """
    df = pd.read_csv(csv_path)

    docs = []
    for _, row in df.iterrows():
        text = (
            f"{row['Company']} ({row['Ticker']}) has a stock price of {row['Price']} USD, "
            f"a P/E ratio of {row['P/E Ratio'] if pd.notna(row['P/E Ratio']) else 'N/A'}, "
            f"and a market capitalization of {row['Market Cap (B)']} billion USD."
        )

        doc = Document(
            page_content=text,
            metadata=row.to_dict()
        )
        docs.append(doc)

    return docs