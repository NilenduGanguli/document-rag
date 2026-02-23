from sentence_transformers import CrossEncoder
from typing import List

class RerankerService:
    def __init__(self):
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            # BAAI/bge-reranker-v2-m3 is a cross-encoder model for reranking
            self._model = CrossEncoder('BAAI/bge-reranker-v2-m3')
        return self._model

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """
        Reranks a list of texts against a query using a cross-encoder model.
        Returns a list of scores corresponding to the input texts.
        """
        if not texts:
            return []
        
        # Create pairs of [query, text]
        pairs = [[query, text] for text in texts]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        return scores.tolist()

reranker_service = RerankerService()
