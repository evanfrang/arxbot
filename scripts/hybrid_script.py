import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from arxbot.retriever import ArxbotRetriever
from arxbot.hybrid_search import hybrid_query

def main():
    df = pd.read_parquet("data/arxiv_HEP_grav.parquet")

    corpus = df["title"].fillna("") + " " + df["abstract"].fillna("")
    tokenized_corpus = [nltk.word_tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    title_to_index = {
        str(t).strip().lower(): i for i, t in enumerate(df["title"])
    }

    retriever = ArxbotRetriever()

    # Query
    query = "Heavy Vectors in Higgs-less models"
    results = hybrid_query(query, retriever, bm25, df, title_to_index, alpha=0.5)

    # Print results
    for r in results:
        print(f"Title: {r['metadata']['title']}")
        print(f"Score: {r['score']:.4f}")
        print("---")

if __name__ == '__main__':
    main()