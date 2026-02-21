from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.domain import ExtractedEntity, SemanticChildChunk, ParsedLayoutSegment, RetrievalAuditLog
from app.schemas.retrieve import RetrievedChunk
from app.services.nlp_service import extract_entities
from app.services.embedding_service import embedding_service
from app.services.reranker_service import reranker_service
import uuid
from typing import List, Dict, Any, Optional

class RetrievalService:
    def __init__(self, db: Session):
        self.db = db

    def parse_query_intent(self, query: str) -> List[Dict[str, str]]:
        # Actual NLP parsing using spaCy and custom regex
        return extract_entities(query)

    def deterministic_bypass(self, entities: List[Dict[str, str]], customer_id: Optional[str]) -> Optional[RetrievedChunk]:
        for entity in entities:
            if entity["type"] == "PASSPORT_NUM":
                # Execute exact match SQL
                query = text("""
                    SELECT 
                        pls.segment_id,
                        pls.doc_id,
                        pls.raw_content,
                        ee.entity_type,
                        ee.entity_value,
                        scc.chunk_id,
                        scc.text_content
                    FROM 
                        extracted_entity ee
                    JOIN 
                        semantic_child_chunk scc ON ee.chunk_id = scc.chunk_id
                    JOIN 
                        parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                    JOIN 
                        raw_document rd ON pls.doc_id = rd.doc_id
                    WHERE 
                        ee.entity_type = 'PASSPORT_NUM' 
                        AND ee.entity_value = :entity_value
                        AND (:customer_id IS NULL OR rd.customer_id = :customer_id)
                    LIMIT 1;
                """)
                result = self.db.execute(query, {"entity_value": entity["value"], "customer_id": customer_id}).fetchone()
                
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
        # Actual embedding generation using BAAI/bge-m3
        query_vector = embedding_service.get_embedding(query)
        
        # Session-Level Tuning: Adjust ef_search for this transaction
        self.db.execute(text("SET LOCAL hnsw.ef_search = 100;"))
        
        # Execute Hybrid Search with RRF
        sql_query = text("""
            WITH dense_search AS (
                SELECT 
                    scc.chunk_id,
                    scc.segment_id,
                    scc.text_content,
                    scc.dense_vector <=> :query_vector::vector AS vector_distance,
                    ROW_NUMBER() OVER (ORDER BY scc.dense_vector <=> :query_vector::vector ASC) AS dense_rank
                FROM 
                    semantic_child_chunk scc
                WHERE 
                    (:customer_id IS NULL OR EXISTS (
                        SELECT 1 FROM parsed_layout_segment pls 
                        JOIN raw_document rd ON pls.doc_id = rd.doc_id 
                        WHERE pls.segment_id = scc.segment_id AND rd.customer_id = :customer_id
                    ))
                ORDER BY 
                    vector_distance ASC
                LIMIT 60
            ),
            sparse_search AS (
                SELECT 
                    scc.chunk_id,
                    scc.segment_id,
                    scc.text_content,
                    ts_rank_cd(to_tsvector('english', scc.text_content), plainto_tsquery('english', :query)) AS sparse_score,
                    ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', scc.text_content), plainto_tsquery('english', :query)) DESC) AS sparse_rank
                FROM 
                    semantic_child_chunk scc
                WHERE 
                    to_tsvector('english', scc.text_content) @@ plainto_tsquery('english', :query)
                    AND (:customer_id IS NULL OR EXISTS (
                        SELECT 1 FROM parsed_layout_segment pls 
                        JOIN raw_document rd ON pls.doc_id = rd.doc_id 
                        WHERE pls.segment_id = scc.segment_id AND rd.customer_id = :customer_id
                    ))
                ORDER BY 
                    sparse_score DESC
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
            "customer_id": customer_id,
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
        if not chunks:
            return []
            
        # Actual Cross-Encoder Reranking using BAAI/bge-reranker-v2-m3
        texts = [chunk.text_content for chunk in chunks]
        scores = reranker_service.rerank(query, texts)
        
        for chunk, score in zip(chunks, scores):
            chunk.score = score
            
        # Filter by dynamic confidence threshold (e.g., > 0.5)
        valid_chunks = [c for c in chunks if c.score > 0.5]
        
        # Graph Traversal Rules
        parent_counts = {}
        for c in valid_chunks:
            parent_counts[c.parent_segment_id] = parent_counts.get(c.parent_segment_id, 0) + 1
            
        final_chunks = []
        processed_parents = set()
        
        for c in valid_chunks:
            # Frequency Rule: If > 2 children share parent, or Semantic Threshold Rule: score > 0.8
            if parent_counts[c.parent_segment_id] > 2 or c.score > 0.8:
                if c.parent_segment_id not in processed_parents:
                    parent = self.db.query(ParsedLayoutSegment).filter(ParsedLayoutSegment.segment_id == c.parent_segment_id).first()
                    if parent:
                        # Drop children, retrieve broader Parent block
                        final_chunks.append(RetrievedChunk(
                            chunk_id=c.chunk_id, # Using the highest scoring child's ID as reference
                            text_content="[PARENT BLOCK RETRIEVED]",
                            score=c.score,
                            parent_segment_id=c.parent_segment_id,
                            parent_content=parent.raw_content
                        ))
                        processed_parents.add(c.parent_segment_id)
            else:
                if c.parent_segment_id not in processed_parents:
                    final_chunks.append(c)
            
        return sorted(final_chunks, key=lambda x: x.score, reverse=True)

    def log_audit(self, query_id: uuid.UUID, decision: str, scores: Dict[str, float], chunks: List[str]):
        audit_log = RetrievalAuditLog(
            query_id=query_id,
            router_decision=decision,
            confidence_scores=scores,
            retrieved_chunks=chunks
        )
        self.db.add(audit_log)
        self.db.commit()
