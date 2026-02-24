import hashlib
import json
import logging
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy imports so the container starts even if fastembed is not yet installed
_sparse_model = None

def _get_sparse_model():
    """Lazy-load the SPLADE sparse embedding model."""
    global _sparse_model
    if _sparse_model is None:
        try:
            from fastembed import SparseTextEmbedding
            from app.core.config import settings
            _sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_EMBEDDING_MODEL_NAME)
        except Exception as exc:
            logger.warning("Could not load sparse embedding model: %s. Sparse vectors will be empty.", exc)
            _sparse_model = None
    return _sparse_model


class EmbeddingService:
    """
    Wraps BAAI/bge-m3 for dense embeddings and Splade_PP_en_v1 for sparse.

    bge-m3 is an asymmetric model: prefix queries with "query: " and passages
    with "passage: " to unlock its full recall potential (~5 % improvement over
    unprefixed encoding).

    Sparse vectors are stored as {str(token_id): weight} JSONB, enabling
    future dot-product sparse retrieval with pgvector.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(self):
        self._model = None
        self._redis = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from app.core.config import settings
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._model

    @property
    def redis(self):
        """Lazy Redis client (used for query-embedding cache)."""
        if self._redis is None:
            try:
                import redis as redis_lib
                from app.core.config import settings
                self._redis = redis_lib.Redis(
                    host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=2, decode_responses=False
                )
                self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    # ── Query embedding (with Redis cache) ────────────────────────────────────

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Returns a dense embedding for a query string.
        Results are cached in Redis for QUERY_EMBEDDING_CACHE_TTL seconds so
        repeated identical queries skip the model inference.
        """
        from app.core.config import settings

        cache_key = f"qemb:{hashlib.sha256(query.encode()).hexdigest()}"

        # Try cache first
        if self.redis is not None:
            try:
                raw = self.redis.get(cache_key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass

        prefixed = self.QUERY_PREFIX + query
        vector = self.model.encode(prefixed, normalize_embeddings=True).tolist()

        if self.redis is not None:
            try:
                self.redis.setex(cache_key, settings.QUERY_EMBEDDING_CACHE_TTL, json.dumps(vector))
            except Exception:
                pass

        return vector

    # ── Passage (document) embeddings ─────────────────────────────────────────

    def get_passage_embeddings(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """
        Returns normalized dense embeddings for a list of passage texts.
        Processing is capped to `batch_size` per inference call to avoid OOM.
        Each text is prefixed with "passage: " before encoding.
        """
        prefixed = [self.PASSAGE_PREFIX + t for t in texts]
        all_embeddings: List[List[float]] = []

        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start : start + batch_size]
            vecs = self.model.encode(batch, normalize_embeddings=True).tolist()
            all_embeddings.extend(vecs)

        return all_embeddings

    # ── Legacy alias (used by retrieval_service before this change) ───────────

    def get_embedding(self, text: str) -> List[float]:
        """Single-text query embedding (backward-compatible helper)."""
        return self.get_query_embedding(text)

    def get_embeddings(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """Backward-compatible batch passage encoder."""
        return self.get_passage_embeddings(texts, batch_size=batch_size)

    # ── Sparse (SPLADE) embeddings ────────────────────────────────────────────

    def get_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Returns SPLADE sparse vectors for a list of texts.
        Each result is a dict {str(token_id): weight} ready for JSONB storage.
        Returns empty dicts if the sparse model is unavailable.
        """
        sparse_model = _get_sparse_model()
        if sparse_model is None:
            return [{} for _ in texts]

        try:
            embeddings = list(sparse_model.embed(texts))
            result = []
            for emb in embeddings:
                # fastembed SparseEmbedding has .indices (array) and .values (array)
                sparse_dict = {
                    str(int(idx)): float(val)
                    for idx, val in zip(emb.indices, emb.values)
                }
                result.append(sparse_dict)
            return result
        except Exception as exc:
            logger.warning("Sparse embedding failed: %s. Using empty sparse vectors.", exc)
            return [{} for _ in texts]


embedding_service = EmbeddingService()
