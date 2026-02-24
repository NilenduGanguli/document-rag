from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID

class RetrieveRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None
    top_k: int = 5
    # Optional caller-supplied hints that extend deterministic bypass entity types
    routing_hints: Optional[List[str]] = None

class RetrievedChunk(BaseModel):
    chunk_id: UUID
    text_content: str
    score: float
    parent_segment_id: UUID
    parent_content: Optional[Dict[str, Any]] = None
    # How this chunk was retrieved: "direct_sql_bypass" | "dense_vector" | "graph_traversal"
    traversal_path: Optional[str] = None

class RetrieveResponse(BaseModel):
    query_id: UUID
    router_decision: str
    retrieved_chunks: List[RetrievedChunk]
    confidence_scores: Dict[str, float]
    # Per-stage wall-clock latencies in milliseconds
    stage_latencies: Optional[Dict[str, float]] = None
    # Retrieval audit metadata: {reranker_model, hybrid_search_triggered,
    # bypass_triggered, graph_traversal_triggered}
    audit_metadata: Optional[Dict[str, Any]] = None
