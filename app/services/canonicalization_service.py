"""
Entity canonicalization service.

After document ingestion, this Celery task deduplicates ExtractedEntity rows
by (entity_type, normalized_value) using exact matching for IDs and fuzzy
matching (rapidfuzz Levenshtein) for names.

Pipeline per entity type:
  1. Load all unlinked ExtractedEntity rows of that type.
  2. For each entity, compute normalized_value (lowercase, stripped).
  3. Exact-match against existing CanonicalEntity rows of the same type.
  4. If no exact match: fuzzy-match (Levenshtein ratio ≥ CANON_FUZZY_THRESHOLD).
  5. If still no match: create a new CanonicalEntity row.
  6. Link ExtractedEntity → CanonicalEntity via canonical_id FK.
  7. Write SAME_ENTITY edges in knowledge_graph_edge between all ExtractedEntity
     rows that share the same canonical_id.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy.dialects.postgresql import insert

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.domain import (
    CanonicalEntity,
    EntityType,
    ExtractedEntity,
    KnowledgeGraphEdge,
    RelationshipType,
)

logger = logging.getLogger(__name__)

# Entity types that use exact matching only (IDs — no fuzzy)
_EXACT_MATCH_TYPES: Set[EntityType] = {
    EntityType.PASSPORT_NUM,
    EntityType.IBAN,
    EntityType.TAX_ID,
    EntityType.COMPANY_REG_NUM,
}

# Minimum rapidfuzz ratio (0–100) for name fuzzy matching
_FUZZY_THRESHOLD = 90


def _normalize(value: str) -> str:
    return value.lower().strip()


def _fuzzy_match(
    value: str,
    candidates: List[Tuple[uuid.UUID, str]],
    threshold: int = _FUZZY_THRESHOLD,
) -> Optional[uuid.UUID]:
    """
    Returns the canonical_id of the best candidate above `threshold`,
    or None if no match found.

    `candidates` is a list of (canonical_id, canonical_value) tuples.
    """
    try:
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz import process as rf_process

        candidate_values = [c[1] for c in candidates]
        result = rf_process.extractOne(
            value,
            candidate_values,
            scorer=Levenshtein.normalized_similarity,
            score_cutoff=threshold / 100.0,
        )
        if result is not None:
            best_value, score, idx = result
            return candidates[idx][0]
    except Exception as exc:
        logger.warning("Fuzzy match failed: %s", exc)
    return None


@celery_app.task
def canonicalize_entities_task(doc_id: str):
    """
    Celery task: canonicalize all ExtractedEntity rows for a given document.
    Should be chained after batch_embed_and_store completes.
    """
    db = SessionLocal()
    try:
        doc_uuid = uuid.UUID(doc_id)

        # Load all unlinked extracted entities for this document
        # (canonical_id IS NULL → not yet canonicalized)
        from sqlalchemy import text
        entity_rows: List[ExtractedEntity] = (
            db.query(ExtractedEntity)
            .join(
                ExtractedEntity.__table__,
                text("""
                    semantic_child_chunk scc
                    ON extracted_entity.chunk_id = scc.chunk_id
                """),
            )
            .filter(ExtractedEntity.canonical_id == None)  # noqa: E711
            .all()
        )

        # Use a direct SQL query to avoid complex ORM joins
        rows = db.execute(
            text("""
                SELECT ee.entity_id, ee.entity_type, ee.entity_value, ee.chunk_id
                FROM extracted_entity ee
                JOIN semantic_child_chunk scc ON ee.chunk_id = scc.chunk_id
                JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                WHERE pls.doc_id = :doc_id
                  AND ee.canonical_id IS NULL
            """),
            {"doc_id": str(doc_uuid)},
        ).fetchall()

        if not rows:
            return {"status": "no_unlinked_entities", "doc_id": doc_id}

        # Group by entity_type for efficient canonicalization
        by_type: Dict[str, list] = {}
        for row in rows:
            by_type.setdefault(row.entity_type, []).append(row)

        new_edges: List[dict] = []
        # Track canonical_id → set of chunk_ids for SAME_ENTITY edge generation
        canonical_chunks: Dict[uuid.UUID, Set[uuid.UUID]] = {}

        for entity_type_str, entity_list in by_type.items():
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                continue

            use_exact_only = entity_type in _EXACT_MATCH_TYPES

            # Load existing canonical entities of this type for matching
            existing: List[CanonicalEntity] = (
                db.query(CanonicalEntity)
                .filter(CanonicalEntity.entity_type == entity_type)
                .all()
            )
            existing_pairs: List[Tuple[uuid.UUID, str]] = [
                (c.canonical_id, c.canonical_value) for c in existing
            ]
            existing_by_value: Dict[str, uuid.UUID] = {
                c.canonical_value: c.canonical_id for c in existing
            }

            for row in entity_list:
                normalized = _normalize(row.entity_value)
                canonical_id: Optional[uuid.UUID] = None

                # Exact match first
                if normalized in existing_by_value:
                    canonical_id = existing_by_value[normalized]

                # Fuzzy match for name-type entities only
                elif not use_exact_only and existing_pairs:
                    canonical_id = _fuzzy_match(normalized, existing_pairs)

                # Create new canonical entity if no match
                if canonical_id is None:
                    canonical_id = uuid.uuid4()
                    new_canon = CanonicalEntity(
                        canonical_id=canonical_id,
                        entity_type=entity_type,
                        canonical_value=normalized,
                    )
                    db.add(new_canon)
                    db.flush()  # ensure it's available for FK linking
                    existing_pairs.append((canonical_id, normalized))
                    existing_by_value[normalized] = canonical_id

                # Link extracted entity to canonical entity
                db.query(ExtractedEntity).filter(
                    ExtractedEntity.entity_id == row.entity_id
                ).update({"canonical_id": canonical_id})

                # Track for SAME_ENTITY edge generation
                if canonical_id not in canonical_chunks:
                    canonical_chunks[canonical_id] = set()
                canonical_chunks[canonical_id].add(uuid.UUID(str(row.chunk_id)))

        # ── Write SAME_ENTITY edges ────────────────────────────────────────────
        for canonical_id, chunk_ids in canonical_chunks.items():
            id_list = list(chunk_ids)
            for i in range(len(id_list)):
                for j in range(i + 1, len(id_list)):
                    edge_id = uuid.uuid5(
                        id_list[i],
                        f"SAME_ENTITY_{canonical_id}_{id_list[j]}",
                    )
                    new_edges.append({
                        "edge_id": edge_id,
                        "source_node": id_list[i],
                        "target_node": id_list[j],
                        "relationship_type": RelationshipType.SAME_ENTITY,
                        "metadata": {"canonical_id": str(canonical_id)},
                    })

        if new_edges:
            stmt = insert(KnowledgeGraphEdge).values(new_edges)
            stmt = stmt.on_conflict_do_nothing(index_elements=["edge_id"])
            db.execute(stmt)

        db.commit()
        return {
            "status": "ok",
            "doc_id": doc_id,
            "entities_processed": len(rows),
            "same_entity_edges_written": len(new_edges),
        }

    except Exception as exc:
        db.rollback()
        logger.exception("canonicalize_entities_task failed for doc %s: %s", doc_id, exc)
        raise
    finally:
        db.close()
