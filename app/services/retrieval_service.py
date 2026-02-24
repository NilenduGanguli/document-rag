"""
Retrieval service — hybrid dense+sparse search with:
  - Deterministic bypass for any entity type in settings.bypass_entity_types
  - Weighted RRF (configurable dense vs. sparse weights)
  - Cross-encoder reranking with timeout + RRF fallback
  - Single-level parent-child graph traversal
  - Multi-hop graph traversal via recursive CTE (P2.6)
  - Per-stage latency tracking (P1.4)
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.domain import (
    ParsedLayoutSegment,
    RetrievalAuditLog,
)
from app.schemas.retrieve import RetrievedChunk
from app.services.embedding_service import embedding_service
from app.services.nlp_service import extract_entities
from app.services.reranker_service import reranker_service


class RetrievalService:
    def __init__(self, db: Session):
        self.db = db

    # ── Stage 1: Query intent / NER ───────────────────────────────────────────

    def parse_query_intent(self, query: str) -> List[Dict]:
        return extract_entities(query)

    # ── Stage 2: Deterministic bypass ─────────────────────────────────────────

    def deterministic_bypass(
        self,
        entities: List[Dict],
        customer_id: Optional[str],
        extra_types: Optional[Set[str]] = None,
    ) -> Optional[RetrievedChunk]:
        """
        Exact-match SQL lookup for any entity whose type is in
        settings.bypass_entity_types (or caller-supplied extra_types).
        Returns the first match, or None if no entity qualifies.
        """
        from app.core.config import settings

        bypass_types = settings.bypass_entity_types | (extra_types or set())

        for entity in entities:
            if entity.get("type") not in bypass_types:
                continue

            result = self.db.execute(
                text("""
                    SELECT
                        pls.segment_id,
                        pls.raw_content,
                        scc.chunk_id,
                        scc.text_content
                    FROM extracted_entity ee
                    JOIN semantic_child_chunk scc ON ee.chunk_id = scc.chunk_id
                    JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                    JOIN raw_document rd ON pls.doc_id = rd.doc_id
                    WHERE ee.entity_type = :entity_type
                      AND ee.entity_value = :entity_value
                      AND (:customer_id IS NULL OR rd.customer_id = :customer_id)
                    LIMIT 1
                """),
                {
                    "entity_type": entity["type"],
                    "entity_value": entity["value"],
                    "customer_id": customer_id,
                },
            ).fetchone()

            if result:
                chunk = RetrievedChunk(
                    chunk_id=result.chunk_id,
                    text_content=result.text_content,
                    score=1.0,
                    parent_segment_id=result.segment_id,
                    parent_content=result.raw_content,
                    traversal_path="direct_sql_bypass",
                )
                return chunk
        return None

    # ── Stage 3: Hybrid dense + sparse search with weighted RRF ───────────────

    def hybrid_search(
        self, query: str, customer_id: Optional[str], top_k: int
    ) -> List[RetrievedChunk]:
        from app.core.config import settings

        query_vector = embedding_service.get_query_embedding(query)
        self.db.execute(text("SET LOCAL hnsw.ef_search = 100;"))

        k = settings.RRF_K
        dense_w = settings.RRF_DENSE_WEIGHT
        sparse_w = 1.0 - dense_w

        results = self.db.execute(
            text("""
                WITH dense_search AS (
                    SELECT
                        scc.chunk_id,
                        scc.segment_id,
                        scc.text_content,
                        ROW_NUMBER() OVER (
                            ORDER BY scc.dense_vector <=> CAST(:query_vector AS vector) ASC
                        ) AS dense_rank
                    FROM semantic_child_chunk scc
                    WHERE (:customer_id IS NULL OR EXISTS (
                        SELECT 1 FROM parsed_layout_segment pls
                        JOIN raw_document rd ON pls.doc_id = rd.doc_id
                        WHERE pls.segment_id = scc.segment_id
                          AND rd.customer_id = :customer_id
                    ))
                    ORDER BY scc.dense_vector <=> CAST(:query_vector AS vector) ASC
                    LIMIT :rrf_limit
                ),
                sparse_search AS (
                    SELECT
                        scc.chunk_id,
                        scc.segment_id,
                        scc.text_content,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank_cd(
                                to_tsvector('english', scc.text_content),
                                plainto_tsquery('english', :query)
                            ) DESC
                        ) AS sparse_rank
                    FROM semantic_child_chunk scc
                    WHERE to_tsvector('english', scc.text_content)
                              @@ plainto_tsquery('english', :query)
                      AND (:customer_id IS NULL OR EXISTS (
                            SELECT 1 FROM parsed_layout_segment pls
                            JOIN raw_document rd ON pls.doc_id = rd.doc_id
                            WHERE pls.segment_id = scc.segment_id
                              AND rd.customer_id = :customer_id
                      ))
                    ORDER BY ts_rank_cd(
                        to_tsvector('english', scc.text_content),
                        plainto_tsquery('english', :query)
                    ) DESC
                    LIMIT :rrf_limit
                )
                SELECT
                    COALESCE(d.chunk_id,    s.chunk_id)    AS chunk_id,
                    COALESCE(d.segment_id,  s.segment_id)  AS segment_id,
                    COALESCE(d.text_content, s.text_content) AS text_content,
                    COALESCE(:dense_w / (:rrf_k + d.dense_rank),  0.0)
                    + COALESCE(:sparse_w / (:rrf_k + s.sparse_rank), 0.0) AS rrf_score
                FROM dense_search d
                FULL OUTER JOIN sparse_search s ON d.chunk_id = s.chunk_id
                ORDER BY rrf_score DESC
                LIMIT :top_k
            """),
            {
                "query_vector": str(query_vector),
                "query": query,
                "customer_id": customer_id,
                "top_k": top_k,
                "rrf_limit": 60,
                "rrf_k": float(k),
                "dense_w": dense_w,
                "sparse_w": sparse_w,
            },
        ).fetchall()

        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                text_content=row.text_content,
                score=row.rrf_score,
                parent_segment_id=row.segment_id,
                traversal_path="dense_vector",
            )
            for row in results
        ]

    # ── Stage 4: Reranking + parent-child graph traversal ─────────────────────

    def rerank_and_traverse(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], float]:
        """
        Cross-encoder reranking with RRF fallback on timeout, followed by
        parent-block promotion when frequency or semantic threshold is met.

        Returns (final_chunks, rerank_ms).
        """
        if not chunks:
            return [], 0.0

        from app.core.config import settings

        rrf_scores = [c.score for c in chunks]
        texts = [c.text_content for c in chunks]

        t0 = time.perf_counter()
        scores = reranker_service.rerank(query, texts, fallback_scores=rrf_scores)
        rerank_ms = (time.perf_counter() - t0) * 1000

        for chunk, score in zip(chunks, scores):
            chunk.score = score

        valid_chunks = [c for c in chunks if c.score > settings.RERANKER_CONFIDENCE_CUTOFF]

        parent_counts: Dict[uuid.UUID, int] = {}
        for c in valid_chunks:
            parent_counts[c.parent_segment_id] = parent_counts.get(c.parent_segment_id, 0) + 1

        final_chunks: List[RetrievedChunk] = []
        processed_parents: Set[uuid.UUID] = set()

        for c in valid_chunks:
            promote = (
                parent_counts[c.parent_segment_id] > settings.GRAPH_FREQUENCY_THRESHOLD
                or c.score > settings.GRAPH_SEMANTIC_THRESHOLD
            )
            if promote:
                if c.parent_segment_id not in processed_parents:
                    parent = self.db.query(ParsedLayoutSegment).filter(
                        ParsedLayoutSegment.segment_id == c.parent_segment_id
                    ).first()
                    if parent:
                        final_chunks.append(
                            RetrievedChunk(
                                chunk_id=c.chunk_id,
                                text_content="[PARENT BLOCK RETRIEVED]",
                                score=c.score,
                                parent_segment_id=c.parent_segment_id,
                                parent_content=parent.raw_content,
                            )
                        )
                        processed_parents.add(c.parent_segment_id)
            else:
                if c.parent_segment_id not in processed_parents:
                    final_chunks.append(c)

        return sorted(final_chunks, key=lambda x: x.score, reverse=True), rerank_ms

    # ── Stage 5: Multi-hop graph traversal (recursive CTE) ────────────────────

    def graph_traversal(
        self,
        chunk_ids: List[uuid.UUID],
        max_depth: Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], float]:
        """
        Walk the knowledge_graph_edge table up to `max_depth` hops starting
        from `chunk_ids`, returning SemanticChildChunk rows reachable via any
        edge type (SAME_ADDRESS, CONTRADICTS, REFERENCES_UBO, etc.).

        Seed nodes are excluded from results.  Returns (extra_chunks, traversal_ms).
        """
        from app.core.config import settings

        t0 = time.perf_counter()

        if not chunk_ids:
            return [], 0.0

        depth = min(max_depth if max_depth is not None else settings.GRAPH_MAX_DEPTH, 4)

        # Build a safe PostgreSQL array literal from validated UUID objects
        arr_literal = "{" + ",".join(str(cid) for cid in chunk_ids) + "}"

        try:
            rows = self.db.execute(
                text("""
                    WITH RECURSIVE graph_walk(node_id, depth, path) AS (
                        SELECT u AS node_id, 0 AS depth, ARRAY[u] AS path
                        FROM unnest(CAST(:start_nodes AS uuid[])) u

                        UNION ALL

                        SELECT
                            kge.target_node,
                            gw.depth + 1,
                            gw.path || kge.target_node
                        FROM knowledge_graph_edge kge
                        JOIN graph_walk gw ON kge.source_node = gw.node_id
                        WHERE gw.depth < :max_depth
                          AND NOT kge.target_node = ANY(gw.path)
                    )
                    SELECT DISTINCT
                        scc.chunk_id,
                        scc.segment_id,
                        scc.text_content
                    FROM graph_walk gw
                    JOIN semantic_child_chunk scc ON gw.node_id = scc.chunk_id
                    WHERE NOT scc.chunk_id = ANY(CAST(:start_nodes AS uuid[]))
                    LIMIT 10
                """),
                {"start_nodes": arr_literal, "max_depth": depth},
            ).fetchall()
        except Exception:
            rows = []

        traversal_ms = (time.perf_counter() - t0) * 1000

        extra_chunks = [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                text_content=row.text_content,
                score=0.0,
                parent_segment_id=row.segment_id,
                traversal_path="graph_traversal",
            )
            for row in rows
        ]
        return extra_chunks, traversal_ms

    # ── Audit logging ─────────────────────────────────────────────────────────

    def log_audit(
        self,
        query_id: uuid.UUID,
        decision: str,
        scores: Dict[str, float],
        chunks: List[str],
        stage_latencies: Optional[Dict[str, float]] = None,
    ) -> None:
        audit_log = RetrievalAuditLog(
            query_id=query_id,
            router_decision=decision,
            confidence_scores=scores,
            retrieved_chunks=chunks,
            stage_latencies=stage_latencies or {},
        )
        self.db.add(audit_log)
        self.db.commit()
