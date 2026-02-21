from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self):
        # BAAI/bge-m3 is a 1024-dimensional embedding model
        self.model = SentenceTransformer('BAAI/bge-m3')

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a dense vector embedding for a single text string.
        """
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of text strings in batch.
        """
        return self.model.encode(texts, normalize_embeddings=True).tolist()

embedding_service = EmbeddingService()
