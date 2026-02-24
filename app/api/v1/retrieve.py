import hashlib
import time
import uuid

import redis
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.schemas.retrieve import RetrieveRequest, RetrieveResponse
from app.services.retrieval_service import RetrievalService

router = APIRouter()

# Shared Redis client for semantic result cache (db=2, same as embedding cache)
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=2, decode_responses=True
    )
    redis_client.ping()
except Exception:
    redis_client = None


def _cache_key(request: RetrieveRequest) -> str:
    hints = ",".join(sorted(request.routing_hints or []))
    raw = f"{request.query}_{request.customer_id}_{request.top_k}_{hints}"
    return f"query:{hashlib.sha256(raw.encode()).hexdigest()}"


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context(request: RetrieveRequest, db: Session = Depends(get_db)):
    t_total_start = time.perf_counter()
    stage_latencies: dict = {}

    # ── Stage 0: Semantic cache check ─────────────────────────────────────────
    cache_key = _cache_key(request)
    t0 = time.perf_counter()
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return RetrieveResponse.model_validate_json(cached)
        except redis.RedisError:
            pass
    stage_latencies["cache_check_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    service = RetrievalService(db)
    query_id = uuid.uuid4()

    # ── Stage 1: Query intent / NER ───────────────────────────────────────────
    t0 = time.perf_counter()
    entities = service.parse_query_intent(request.query)
    stage_latencies["entity_extract_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── Stage 2: Deterministic bypass ─────────────────────────────────────────
    t0 = time.perf_counter()
    extra_types = set(request.routing_hints) if request.routing_hints else None
    exact_match = service.deterministic_bypass(entities, request.customer_id, extra_types)
    stage_latencies["deterministic_bypass_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if exact_match:
        stage_latencies["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        service.log_audit(
            query_id=query_id,
            decision="metadata_exact_match",
            scores={"exact_match": 1.0},
            chunks=[str(exact_match.chunk_id)],
            stage_latencies=stage_latencies,
        )
        audit_metadata = {
            "reranker_model": settings.RERANKER_MODEL_NAME,
            "hybrid_search_triggered": False,
            "bypass_triggered": True,
            "graph_traversal_triggered": False,
        }
        response = RetrieveResponse(
            query_id=query_id,
            router_decision="metadata_exact_match",
            retrieved_chunks=[exact_match],
            confidence_scores={"exact_match": 1.0},
            stage_latencies=stage_latencies,
            audit_metadata=audit_metadata,
        )
        _cache_store(cache_key, response)
        return response

    # ── Stage 3: Hybrid dense + sparse search ─────────────────────────────────
    t0 = time.perf_counter()
    hybrid_results = service.hybrid_search(request.query, request.customer_id, request.top_k)
    stage_latencies["hybrid_search_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ── Stage 4: Reranking + parent-child traversal ───────────────────────────
    reranked, rerank_ms = service.rerank_and_traverse(request.query, hybrid_results)
    stage_latencies["reranker_ms"] = round(rerank_ms, 2)

    # ── Stage 5: Multi-hop graph traversal ────────────────────────────────────
    t0 = time.perf_counter()
    seed_ids = [c.chunk_id for c in reranked]
    graph_extras, traversal_ms = service.graph_traversal(seed_ids)
    stage_latencies["graph_traversal_ms"] = round(traversal_ms, 2)

    # Merge graph-traversed chunks (no duplicates) behind reranked results
    existing_ids = {c.chunk_id for c in reranked}
    for extra in graph_extras:
        if extra.chunk_id not in existing_ids:
            reranked.append(extra)
            existing_ids.add(extra.chunk_id)

    stage_latencies["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)

    avg_score = (
        sum(r.score for r in reranked) / len(reranked) if reranked else 0.0
    )
    confidence_scores = {"reranker_avg": avg_score}

    service.log_audit(
        query_id=query_id,
        decision="hybrid_search_rrf",
        scores=confidence_scores,
        chunks=[str(r.chunk_id) for r in reranked],
        stage_latencies=stage_latencies,
    )

    audit_metadata = {
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "hybrid_search_triggered": True,
        "bypass_triggered": False,
        "graph_traversal_triggered": len(graph_extras) > 0,
    }

    response = RetrieveResponse(
        query_id=query_id,
        router_decision="hybrid_search_rrf",
        retrieved_chunks=reranked,
        confidence_scores=confidence_scores,
        stage_latencies=stage_latencies,
        audit_metadata=audit_metadata,
    )
    _cache_store(cache_key, response)
    return response


def _cache_store(key: str, response: RetrieveResponse) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(key, settings.QUERY_CACHE_TTL, response.model_dump_json())
    except redis.RedisError:
        pass
