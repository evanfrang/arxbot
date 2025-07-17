import chromadb
from sentence_transformers import SentenceTransformer
from arxbot.embedding import Specter2Embedder
import numpy as np


class ArxbotRetriever:
    def __init__(
        self,
        persist_dir="chroma_db",
        collection_name="arxbot_abstracts",
        top_k=5,
        embedding="specter2",  # or a model instance
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(name=collection_name)
        self.top_k = top_k

        self.model = self._load_embedder(embedding)

    def _load_embedder(self, embedding):
        if isinstance(embedding, str):
            if embedding.lower() == "specter2":
                return Specter2Embedder()
            elif embedding.lower() == "bge":
                return SentenceTransformer("BAAI/bge-base-en-v1.5")
            elif embedding.lower() == "mini":
                return SentenceTransformer("all-MiniLM-L6-v2")
            else:
                raise ValueError(f"Unknown embedding model name: {embedding}")
        else:
            return embedding  # custom instance passed in

    def embed_query(self, query: str):
        return self.model.encode([query])[0].tolist()

    def query(self, query: str):
        query_emb = self.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_all_embeddings(self):
        """
        Returns:
            embeddings: np.ndarray of shape (N, D)
            ids: list of document IDs
            metadatas: list of metadata dictionaries
        """
        results = self.collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(results["embeddings"])
        ids = results["ids"]
        metadatas = results["metadatas"]
        return embeddings, ids, metadatas