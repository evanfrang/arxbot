from arxbot.embedding import embed_texts_chunked, load_all_embeddings
from arxbot.ingestion import load_arxiv_data

abstracts = load_arxiv_data("data/arxiv_100k_processed.parquet")

embed_texts_chunked(abstracts, save_dir='embeddings', chunk_size=500, batch_size=16)

embeddings = load_all_embeddings('embeddings')
print("Loaded embeddings shape:", embeddings.shape)