from sklearn.preprocessing import minmax_scale
import numpy as np
import nltk


def hybrid_query(query, retriever, bm25, df, title_to_index, top_k=5, alpha=0.5):
    results = retriever.query(query)
    vector_docs = results["documents"][0]
    vector_metas = results["metadatas"][0]
    vector_scores = 1 - np.array(results["distances"][0])  # convert distance to similarity

    tokenized_query = nltk.word_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    hybrid = []
    for i, meta in enumerate(vector_metas):
        title = meta.get("title", "").strip()
        idx = title_to_index.get(title)
        bm25_score = bm25_scores[idx] if idx is not None else 0.0
        hybrid_score = alpha * vector_scores[i] + (1 - alpha) * bm25_score
        hybrid.append((hybrid_score, vector_docs[i], meta))

    top = sorted(hybrid, reverse=True)[:top_k]
    return [{"score": s, "document": d, "metadata": m} for s, d, m in top]