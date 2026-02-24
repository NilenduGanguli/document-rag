"""
GraphRepository abstraction — decouples graph operations from the
PostgreSQL-specific implementation so the backing store can be swapped
(e.g., to Neo4j) by changing a config value, not rewriting service code.

Current implementation: PostgresGraphRepository (knowledge_graph_edge table).
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.domain import KnowledgeGraphEdge, RelationshipType


class GraphRepository(ABC):
    """Abstract interface for graph operations."""

    @abstractmethod
    def add_edges(self, edges: List[Dict[str, Any]]) -> None:
        """
        Bulk-insert graph edges.  Each dict must contain:
          edge_id, source_node, target_node, relationship_type, metadata (optional).
        Implementations should use ON CONFLICT DO NOTHING semantics.
        """

    @abstractmethod
    def get_neighbors(
        self,
        node_id: uuid.UUID,
        relationship_types: Optional[List[RelationshipType]] = None,
    ) -> List[uuid.UUID]:
        """Return all direct neighbour node_ids of `node_id`."""

    @abstractmethod
    def multi_hop_traverse(
        self,
        start_nodes: List[uuid.UUID],
        max_depth: int = 2,
    ) -> List[uuid.UUID]:
        """
        Return all node_ids reachable from `start_nodes` within `max_depth` hops,
        excluding the seed nodes themselves.
        """

    @abstractmethod
    def get_edges_for_node(
        self,
        node_id: uuid.UUID,
        relationship_types: Optional[List[RelationshipType]] = None,
    ) -> List[KnowledgeGraphEdge]:
        """Return all edges where `node_id` is source or target."""


class PostgresGraphRepository(GraphRepository):
    """
    PostgreSQL implementation backed by the `knowledge_graph_edge` table.

    Multi-hop traversal uses a recursive CTE — no external graph DB required.
    """

    def __init__(self, db: Session):
        self._db = db

    def add_edges(self, edges: List[Dict[str, Any]]) -> None:
        if not edges:
            return
        from sqlalchemy.dialects.postgresql import insert

        stmt = insert(KnowledgeGraphEdge).values(edges)
        stmt = stmt.on_conflict_do_nothing(index_elements=["edge_id"])
        self._db.execute(stmt)
        self._db.commit()

    def get_neighbors(
        self,
        node_id: uuid.UUID,
        relationship_types: Optional[List[RelationshipType]] = None,
    ) -> List[uuid.UUID]:
        q = self._db.query(KnowledgeGraphEdge).filter(
            KnowledgeGraphEdge.source_node == node_id
        )
        if relationship_types:
            q = q.filter(KnowledgeGraphEdge.relationship_type.in_(relationship_types))
        return [e.target_node for e in q.all()]

    def multi_hop_traverse(
        self,
        start_nodes: List[uuid.UUID],
        max_depth: int = 2,
    ) -> List[uuid.UUID]:
        if not start_nodes:
            return []

        depth = min(max_depth, 4)
        arr_literal = "{" + ",".join(str(n) for n in start_nodes) + "}"

        try:
            rows = self._db.execute(
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
                    SELECT DISTINCT node_id
                    FROM graph_walk
                    WHERE NOT node_id = ANY(CAST(:start_nodes AS uuid[]))
                """),
                {"start_nodes": arr_literal, "max_depth": depth},
            ).fetchall()
            return [row.node_id for row in rows]
        except Exception:
            return []

    def get_edges_for_node(
        self,
        node_id: uuid.UUID,
        relationship_types: Optional[List[RelationshipType]] = None,
    ) -> List[KnowledgeGraphEdge]:
        from sqlalchemy import or_

        q = self._db.query(KnowledgeGraphEdge).filter(
            or_(
                KnowledgeGraphEdge.source_node == node_id,
                KnowledgeGraphEdge.target_node == node_id,
            )
        )
        if relationship_types:
            q = q.filter(KnowledgeGraphEdge.relationship_type.in_(relationship_types))
        return q.all()


def get_graph_repository(db: Session) -> GraphRepository:
    """
    Factory function — returns the configured GraphRepository implementation.
    Extend this to support Neo4j by reading a GRAPH_BACKEND setting.
    """
    return PostgresGraphRepository(db)
