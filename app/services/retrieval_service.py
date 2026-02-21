from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.domain import ExtractedEntity, SemanticChildChunk, ParsedLayoutSegment, RetrievalAuditLog
from app.schemas.retrieve import RetrievedChunk
import uuid
from typing import List, Dict, Any, Optional

class RetrievalService:
    def __init__(self, db: Session):
        self.db = db

    def parse_query_intent(self, query: str) -> List[Dict[str, str]]:
        # Mock NLP parsing (e.g., spaCy)
        # In a real scenario, this would call an NLP service
        entities = []
        if "Passport" in query or "passport" in query:
            # Extract passport number using regex or NER
            import re
            match = re.search(r'[A-Z0-9]{5,9}', query)
            if match:
                entities.append({"type": "PASSPORT_NUM", "value": match.group(0)})
        return entities

    def deterministic_bypass(self, entities: List[Dict[str, str]], customer_id: Optional[str]) -> Optional[RetrievedChunk]:
        for entity in entities:
            if entity["type"] == "PASSPORT_NUM":
                # Execute exact match SQL
                query = text("""
                    SELECT 
                        pls.segment_id,
                        pls.raw_content,
                        scc.chunk_id,
                        scc.text_content,
                        ee.entity_value
                    FROM 
                        extracted_entity ee
                    JOIN 
                        semantic_child_chunk scc ON ee.chunk_id = scc.chunk_id
                    JOIN 
                        parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                    WHERE 
                        ee.entity_type = 'PASSPORT_NUM' 
                        AND ee.entity_value = :entity_value
                    LIMIT 1;
                """)
                result = self.db.execute(query, {"entity_value": entity["value"]}).fetchone()
                
                if result:
                    return RetrievedChunk(
                        chunk_id=result.chunk_id,
                        text_content=result.text_content,
                        score=1.0,
                        parent_segment_id=result.segment_id,
                        parent_content=result.raw_content
                    )
        return None

    def hybrid_search(self, query: str, customer_id: Optional[str], top_k: int) -> List[RetrievedChunk]:
        # Mock embedding generation
        # In a real scenario, this would call an embedding service
        query_vector = [0.0] * 1024 # Placeholder for 1024-dim vector
        
        # Execute Hybrid Search with RRF
        sql_query = text("""
            WITH dense_search AS (
                SELECT 
                    chunk_id,
                    segment_id,
                    text_content,
                    dense_vector <=> :query_vector::vector AS distance,
                    RANK() OVER (ORDER BY dense_vector <=> :query_vector::vector) AS dense_rank
                FROM 
                    semantic_child_chunk
                LIMIT 60
            ),
            sparse_search AS (
                SELECT 
                    chunk_id,
                    segment_id,
                    text_content,
                    ts_rank_cd(to_tsvector('english', text_content), plainto_tsquery('english', :query)) AS rank,
                    RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('english', text_content), plainto_tsquery('english', :query)) DESC) AS sparse_rank
                FROM 
                    semantic_child_chunk
                WHERE 
                    to_tsvector('english', text_content) @@ plainto_tsquery('english', :query)
                LIMIT 60
            )
            SELECT 
                COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
                COALESCE(d.segment_id, s.segment_id) AS segment_id,
                COALESCE(d.text_content, s.text_content) AS text_content,
                COALESCE(1.0 / (60 + d.dense_rank), 0.0) + COALESCE(1.0 / (60 + s.sparse_rank), 0.0) AS rrf_score
            FROM 
                dense_search d
            FULL OUTER JOIN 
                sparse_search s ON d.chunk_id = s.chunk_id
            ORDER BY 
                rrf_score DESC
            LIMIT :top_k;
        """)
        
        results = self.db.execute(sql_query, {
            "query_vector": str(query_vector),
            "query": query,
            "top_k": top_k
        }).fetchall()
        
        chunks = []
        for row in results:
            chunks.append(RetrievedChunk(
                chunk_id=row.chunk_id,
                text_content=row.text_content,
                score=row.rrf_score,
                parent_segment_id=row.segment_id
            ))
        return chunks

    def rerank_and_traverse(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        # Mock Cross-Encoder Reranking
        # In a real scenario, this would call a reranker model
        for chunk in chunks:
            chunk.score *= 0.9 # Mock reranking adjustment
            
            # Graph Traversal: Fetch parent content if score is high
            if chunk.score > 0.01: # Threshold
                parent = self.db.query(ParsedLayoutSegment).filter(ParsedLayoutSegment.segment_id == chunk.parent_segment_id).first()
                if parent:
                    chunk.parent_content = parent.raw_content
                    
        return sorted(chunks, key=lambda x: x.score, reverse=True)

    def log_audit(self, query_id: uuid.UUID, decision: str, scores: Dict[str, float], chunks: List[str]):
        audit_log = RetrievalAuditLog(
            query_id=query_id,
            router_decision=decision,
            confidence_scores=scores,
            retrieved_chunks=chunks
        )
        self.db.add(audit_log)
        self.db.commit()
