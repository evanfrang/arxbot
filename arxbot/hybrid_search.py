from sklearn.preprocessing import minmax_scale
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict



def get_top_dense_results(query, retriever):

    results = retriever.query(query)

    return [
        {"score": 1 - dist, "metadata": meta}
        for dist, meta in zip(results["distances"][0], results["metadatas"][0])
    ]

def get_top_bm25_results(query: str, df, bm25_model, top_k=50):

    query_tokens = query.lower().split()

    # Get BM25 scores
    scores = bm25_model.get_scores(query_tokens)

    # Get top-k indices
    top_k_indices = np.argsort(scores)[::-1][:top_k]

    # Build results in the same format as dense
    results = [
        {
            "score": scores[i],
            "metadata": df.iloc[i].to_dict()
        }
        for i in top_k_indices
    ]

    return results

def rrf_fuse(bm25_results, dense_results, df, k=60, limit=5):
    rrf_scores = defaultdict(float)
    rrf_ranks = defaultdict(float)

    # Helper to get row index from metadata, fallback to None
    def get_doc_idx(metadata):
        # Try 'id' in metadata
        if 'id' in metadata:
            return metadata['id']
        # If no id, try to find index by unique title (if titles unique)
        if 'title' in metadata:
            title = metadata['title'].strip().lower()
            matches = df.index[df['title'].str.lower() == title].tolist()
            if matches:
                return matches[0]
        # fallback
        return None

    # Accumulate RRF scores
    for method_results in [bm25_results, dense_results]:
        for rank, result in enumerate(method_results):
            doc_idx = get_doc_idx(result['metadata'])
            if doc_idx is None:
                continue
            rrf_scores[doc_idx] += 1 / (k + rank)
            #rrf_ranks[doc_idx] = rank

    # Sort descending by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    fused_results = []
    for doc_idx, score in sorted_docs:
        metadata = df.iloc[doc_idx].to_dict()
        fused_results.append({
            "score": score,
            "metadata": metadata
        })

    return fused_results