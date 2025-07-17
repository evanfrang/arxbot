import pandas as pd
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from rank_bm25 import BM25Okapi
from arxbot.hybrid_search import get_top_bm25_results
from arxbot.rag_ollama import rag_with_bm25


def main():
    df = pd.read_parquet("data/arxiv_HEP_grav.parquet").reset_index()
    
    corpus = df["title"].fillna("") + " " + df["abstract"].fillna("")
    tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    while True:
        user_query = input("\nAsk a question (or 'quit'): ")
        if user_query.strip().lower() == "quit":
            break

        bm25_results = get_top_bm25_results(user_query, df, bm25, top_k=5)

        # RAG
        answer = rag_with_bm25(user_query, bm25_results)
        print("\nOllama:\n", answer)

if __name__ == '__main__':
    main()