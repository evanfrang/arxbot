import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once, use GPU if available
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # 'cuda' uses your GPU

def embed_texts_chunked(texts, save_dir='embeddings', chunk_size=500, batch_size=16):
    """
    Embed texts in chunks and save embeddings incrementally to disk.

    Args:
        texts (list of str): List of documents to embed.
        save_dir (str): Directory to save chunked embeddings.
        chunk_size (int): Number of texts to embed per chunk.
        batch_size (int): Batch size for model.encode().

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    total = len(texts)
    num_chunks = (total + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_path = os.path.join(save_dir, f'embeddings_chunk_{chunk_idx}.npy')
        if os.path.exists(chunk_path):
            print(f"Chunk {chunk_idx} already exists, skipping...")
            continue

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk_texts = texts[start:end]

        print(f"Embedding chunk {chunk_idx + 1}/{num_chunks}, texts {start} to {end}...")
        embeddings = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

        np.save(chunk_path, embeddings)
        print(f"Saved chunk {chunk_idx} embeddings to {chunk_path}")


def load_all_embeddings(save_dir='embeddings'):
    """
    Load all chunked embeddings and stack them into a single numpy array.

    Args:
        save_dir (str): Directory where embeddings are saved.

    Returns:
        np.ndarray: All embeddings stacked.
    """
    files = sorted(f for f in os.listdir(save_dir) if f.startswith('embeddings_chunk_') and f.endswith('.npy'))
    all_embeddings = []
    for f in files:
        path = os.path.join(save_dir, f)
        print(f"Loading {path}")
        chunk_embeds = np.load(path)
        all_embeddings.append(chunk_embeds)

    return np.vstack(all_embeddings)
