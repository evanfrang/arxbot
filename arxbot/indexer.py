import os
import numpy as np
import pandas as pd
import chromadb

def load_embeddings_chunks(emb_dir='embeddings'):
    """
    Load all .npy embedding chunks and stack into a single array.
    """
    chunk_files = sorted([
        f for f in os.listdir(emb_dir)
        if f.startswith('embeddings_chunk_') and f.endswith('.npy')
    ])
    all_embeddings = []
    for file in chunk_files:
        path = os.path.join(emb_dir, file)
        emb = np.load(path)
        all_embeddings.append(emb)
    return np.vstack(all_embeddings)

def add_in_batches(collection, embeddings, documents, metadatas, batch_size=5000):
    total = len(embeddings)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        print(f"Adding batch {start_idx} to {end_idx}...")
        collection.add(
            embeddings=embeddings[start_idx:end_idx].tolist(),
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            ids=[f"doc_{i}" for i in range(start_idx, end_idx)]
)

def build_chroma_index(
    metadata_parquet='data/arxiv_HEP_grav.parquet',
    emb_dir='embeddings',
    persist_dir='chroma_db',
    collection_name='arxbot_abstracts'
):
    # Load metadata
    print(f"Loading metadata from {metadata_parquet}...")
    df = pd.read_parquet(metadata_parquet)
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} rows.")

    # Load embeddings
    embeddings = load_embeddings_chunks(emb_dir)
    assert len(embeddings) == len(df), "Embedding count does not match metadata row count!"

    # Set up Chroma client
    client = chromadb.PersistentClient(path="chroma_db")

    # Create or get the collection
    collection = client.get_or_create_collection(name=collection_name)

    # Insert documents
    print("Adding documents to ChromaDB...")
    add_in_batches(
        collection,
        embeddings,
        df['abstract'].tolist(),
        df.to_dict(orient='records'),
        batch_size=5000
    )

    print(f"✅ Chroma index created and saved to {persist_dir}")