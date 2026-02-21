from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.retrieve import RetrieveRequest, RetrieveResponse
from app.services.retrieval_service import RetrievalService
import uuid

router = APIRouter()

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context(request: RetrieveRequest, db: Session = Depends(get_db)):
    service = RetrievalService(db)
    
    # 1. Query Intent & NER Parsing
    entities = service.parse_query_intent(request.query)
    
    # 2. Deterministic Bypass
    exact_match = service.deterministic_bypass(entities, request.customer_id)
    
    if exact_match:
        # Audit Logging for Deterministic Bypass
        service.log_audit(
            query_id=uuid.uuid4(),
            decision="metadata_exact_match",
            scores={"exact_match": 1.0},
            chunks=[str(exact_match.chunk_id)]
        )
        return RetrieveResponse(
            query_id=uuid.uuid4(),
            router_decision="metadata_exact_match",
            retrieved_chunks=[exact_match],
            confidence_scores={"exact_match": 1.0}
        )
    
    # 3. Hybrid Search (Dense + Sparse RRF)
    hybrid_results = service.hybrid_search(request.query, request.customer_id, request.top_k)
    
    # 4. Cross-Encoder Reranking & Graph Traversal
    reranked_results = service.rerank_and_traverse(request.query, hybrid_results)
    
    # 5. Audit Logging for Hybrid Search
    service.log_audit(
        query_id=uuid.uuid4(),
        decision="hybrid_search_rrf",
        scores={"reranker_avg": sum(r.score for r in reranked_results) / len(reranked_results) if reranked_results else 0},
        chunks=[str(r.chunk_id) for r in reranked_results]
    )
    
    return RetrieveResponse(
        query_id=uuid.uuid4(),
        router_decision="hybrid_search_rrf",
        retrieved_chunks=reranked_results,
        confidence_scores={"reranker_avg": sum(r.score for r in reranked_results) / len(reranked_results) if reranked_results else 0}
    )
