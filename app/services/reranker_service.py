import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Single shared executor for reranker inference calls
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")


class RerankerService:
    """
    Cross-encoder reranker wrapper with configurable timeout and RRF fallback.

    If the model inference takes longer than `RERANKER_TIMEOUT_S` seconds (e.g.
    due to overloaded GPU / large candidate set), the method falls back to the
    provided `fallback_scores` so the pipeline always returns a ranked result.
    """

    def __init__(self):
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from app.core.config import settings
            self._model = CrossEncoder(settings.RERANKER_MODEL_NAME)
        return self._model

    def rerank(
        self,
        query: str,
        texts: List[str],
        fallback_scores: Optional[List[float]] = None,
        timeout: Optional[float] = None,
    ) -> List[float]:
        """
        Reranks `texts` against `query` using the cross-encoder.

        Args:
            query: The user query string.
            texts: Candidate passage strings to score.
            fallback_scores: Scores to return if the model times out or fails.
                             If omitted, a list of zeros is returned on failure.
            timeout: Override the default RERANKER_TIMEOUT_S for this call.

        Returns:
            List of float scores in [0, 1] corresponding to `texts`.
        """
        if not texts:
            return []

        from app.core.config import settings
        effective_timeout = timeout if timeout is not None else settings.RERANKER_TIMEOUT_S
        _fallback = fallback_scores if fallback_scores is not None else [0.0] * len(texts)

        pairs = [[query, text] for text in texts]

        future = _executor.submit(self.model.predict, pairs)
        try:
            scores = future.result(timeout=effective_timeout)
            return scores.tolist()
        except FuturesTimeoutError:
            logger.warning(
                "Reranker timed out after %.1f s for query %r — using RRF fallback scores.",
                effective_timeout,
                query[:80],
            )
            return _fallback
        except Exception as exc:
            logger.error("Reranker inference failed: %s — using fallback scores.", exc)
            return _fallback


reranker_service = RerankerService()
