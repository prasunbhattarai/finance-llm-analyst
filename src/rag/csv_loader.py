import pandas as pd
from langchain.schema import Document

def csv_to_docs(csv_path):
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
