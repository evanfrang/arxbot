from sklearn.preprocessing import minmax_scale
import numpy as np
import nltk


def hybrid_query(query, retriever, bm25, df, title_to_index, top_k=5, alpha=0.5):
    results = retriever.query(query)
    vector_docs = results["documents"][0]
    vector_metas = results["metadatas"][0]
    vector_scores_raw = 1 - np.array(results["distances"][0])  # convert distance to similarity
    vector_scores = minmax_scale(vector_scores_raw)

    tokenized_query = nltk.word_tokenize(query)
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    bm25_scores = minmax_scale(bm25_scores_raw)

    hybrid_results = []
    for i, (meta, score) in enumerate(zip(vector_metas, vector_scores)):
        title = meta.get("title", "").strip().lower()
        idx = title_to_index.get(title)

        if idx is None:
            print(f"Title not found in title_to_index: '{title}'")
            continue
        bm25_score = bm25_scores[idx]
        hybrid_score = alpha * score + (1 - alpha) * bm25_score
        hybrid_results.append((hybrid_score, vector_docs[i], meta))
    
    top = sorted(hybrid_results, key=lambda x: x[0], reverse=True)[:top_k]
    return [{"score": s, "document": d, "metadata": m} for s, d, m in top]