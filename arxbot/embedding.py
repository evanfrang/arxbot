import os
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class Specter2Embedder:
    def __init__(self, model_name="allenai/specter2_base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(texts, 
                padding=True, 
                truncation=True,
                max_length=512, 
                return_tensors="pt"
            )
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            return embeddings.cpu().numpy()

# Load model once, use GPU if available
#model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')  # 'cuda' uses your GPU

def embed_texts_chunked(texts, save_dir='embeddings', chunk_size=50, batch_size=16):
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

    embedder = Specter2Embedder()
    
    for chunk_idx in range(num_chunks):
        chunk_path = os.path.join(save_dir, f'embeddings_chunk_{chunk_idx}.npy')
        if os.path.exists(chunk_path):
            print(f"Chunk {chunk_idx} already exists, skipping...")
            continue

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk_texts = texts[start:end]

        print(f"Embedding chunk {chunk_idx + 1}/{num_chunks}, texts {start} to {end}...")
        #embeddings = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        
        texts_to_embed = [
            str((item.get('title', '') + " " + item.get('abstract', '')).strip())
            for item in chunk_texts
        ]
        max_len = max(len(t) for t in texts_to_embed)
        
        print(f"Max text length in chunk {chunk_idx}: {max_len}")
        embeddings = embedder.encode(texts_to_embed) 
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

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
