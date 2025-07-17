import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from arxbot.retriever import ArxbotRetriever
from arxbot.hybrid_search import rrf_fuse, get_top_bm25_results, get_top_dense_results
import matplotlib.pyplot as plt
import umap
import numpy as np


def plot_embeddings(doc_embeddings, query_embedding, titles, top_k_indices):
    all_embeddings = np.vstack([doc_embeddings, query_embedding])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], alpha=0.4, label="Corpus")

    # Highlight top-k
    if top_k_indices is not None:
        top_k_points = embeddings_2d[top_k_indices]
        plt.scatter(top_k_points[:, 0], top_k_points[:, 1], color="orange", label="Top-K")

    # Plot query
    query_point = embeddings_2d[-1]
    plt.scatter(query_point[0], query_point[1], color='red', label='Query', marker='x', s=100)

    plt.legend()
    plt.title("UMAP of Specter Embeddings + Query")
    plt.savefig(f"results/umap_web.png")

def main():
    df = pd.read_parquet("data/arxiv_HEP_grav.parquet").reset_index()
    
    corpus = df["title"].fillna("") + " " + df["abstract"].fillna("")
    tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    title_to_index = {
        str(t).strip().lower(): i for i, t in enumerate(df["title"])
    }

    retriever = ArxbotRetriever()

    # Query
    query = "AdS black hole background. The quantum fields modeling the Hawking".lower()
    dense_results = get_top_dense_results(query, retriever)
    bm25_results = get_top_bm25_results(query, df, bm25)
    results = rrf_fuse(bm25_results, dense_results, df)

    # Print results
    for r in results:
        print(f"scores: {r['score']}")
        print(f"meta: {r['metadata']}")
        print("---")

    for r in dense_results:
        print(f"scores: {r['score']}")
        print(f"meta: {r['metadata']}")
        print("---")

    for r in bm25_results[:10]:
        print(f"scores: {r['score']}")
        print(f"meta: {r['metadata']}")
        print("---")


    doc_embeddings, ids, metadatas = retriever.get_all_embeddings()
    query_embedding = np.array(retriever.embed_query(query)).reshape(1, -1)

    # Get indices of returned docs (match title to df)
    top_titles = [x['metadata']['title'].strip().lower() for x in results]
    top_k_indices = [title_to_index.get(t) for t in top_titles if title_to_index.get(t) is not None]

    plot_embeddings(doc_embeddings, query_embedding, df["title"], top_k_indices)

if __name__ == '__main__':
    main()