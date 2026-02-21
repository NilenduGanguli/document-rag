from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID

class RetrieveRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None
    top_k: int = 5

class RetrievedChunk(BaseModel):
    chunk_id: UUID
    text_content: str
    score: float
    parent_segment_id: UUID
    parent_content: Optional[Dict[str, Any]] = None

class RetrieveResponse(BaseModel):
    query_id: UUID
    router_decision: str
    retrieved_chunks: List[RetrievedChunk]
    confidence_scores: Dict[str, float]
