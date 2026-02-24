import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Enum, ForeignKey, JSON, Float, Integer, Index, text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base
import enum


# ── Enumerations ──────────────────────────────────────────────────────────────

class FileType(str, enum.Enum):
    PDF = "PDF"
    PNG = "PNG"
    JPEG = "JPEG"
    TIFF = "TIFF"
    DOCX = "DOCX"
    XLSX = "XLSX"


class EntityType(str, enum.Enum):
    # Standard KYC entities (spaCy)
    PERSON = "PERSON"
    ORG = "ORG"
    DATE = "DATE"
    ADDRESS = "ADDRESS"
    # Document-specific
    PASSPORT_NUM = "PASSPORT_NUM"
    # GLiNER-sourced KYC entities
    UBO = "UBO"
    BENEFICIAL_OWNER = "BENEFICIAL_OWNER"
    COMPANY_REG_NUM = "COMPANY_REG_NUM"
    TAX_ID = "TAX_ID"
    IBAN = "IBAN"
    SANCTION_ENTRY = "SANCTION_ENTRY"


class RelationshipType(str, enum.Enum):
    CHILD_OF = "CHILD_OF"          # SemanticChildChunk → ParsedLayoutSegment
    SAME_ADDRESS = "SAME_ADDRESS"   # Two entities sharing a normalized address
    CONTRADICTS = "CONTRADICTS"     # DATE / fact conflicts between documents
    REFERENCES_UBO = "REFERENCES_UBO"  # Entity references a UBO
    SAME_ENTITY = "SAME_ENTITY"    # Two ExtractedEntity rows canonicalized to same entity


# ── Core Document Models ──────────────────────────────────────────────────────

class RawDocument(Base):
    __tablename__ = "raw_document"

    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(String, index=True, nullable=False)
    file_type = Column(Enum(FileType), nullable=False)
    storage_uri = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    # Set to non-null once the Celery pipeline has finished successfully
    processed_at = Column(DateTime(timezone=True), nullable=True)
    # Stores processing_directives from the ingest request: {force_ocr, ocr_provider}
    # NOTE: production deployments require an Alembic migration to add this column
    processing_directives = Column(JSONB, nullable=True)
    # Written by batch_embed_and_store after pipeline completes:
    # {segments: int, chunks: int, edges: int, entities: {PERSON: 3, ORG: 1, ...}, error: str|null}
    # NOTE: production deployments require an Alembic migration to add this column
    processing_summary = Column(JSONB, nullable=True)

    segments = relationship("ParsedLayoutSegment", back_populates="document")


class ParsedLayoutSegment(Base):
    __tablename__ = "parsed_layout_segment"

    segment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("raw_document.doc_id"), nullable=False)
    # Structured JSONB: {type, content, raw_text, metadata}
    # Tables stored as {"rows": [...], "headers": [...]} — not raw HTML
    raw_content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    document = relationship("RawDocument", back_populates="segments")
    chunks = relationship("SemanticChildChunk", back_populates="segment")


class SemanticChildChunk(Base):
    __tablename__ = "semantic_child_chunk"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    segment_id = Column(UUID(as_uuid=True), ForeignKey("parsed_layout_segment.segment_id"), nullable=False)
    # Plain text used for BM25 full-text search (GIN index below)
    text_content = Column(Text, nullable=False)
    # 1024-dim BAAI/bge-m3 dense vector (HNSW index below)
    dense_vector = Column(Vector(1024))
    # SPLADE sparse vector: {token_id_str: weight_float} populated by fastembed
    sparse_vector = Column(JSONB)
    # Chunk-level quality / provenance metadata:
    # {char_count, content_type, position_in_segment, has_table, embed_prefix}
    chunk_metadata = Column(JSONB)

    segment = relationship("ParsedLayoutSegment", back_populates="chunks")
    entities = relationship("ExtractedEntity", back_populates="chunk")

    __table_args__ = (
        # HNSW index: m=24 (connections per node), ef_construction=100 (build accuracy)
        Index(
            "idx_semantic_child_vector",
            "dense_vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 24, "ef_construction": 100},
            postgresql_ops={"dense_vector": "vector_cosine_ops"},
        ),
        # GIN index for PostgreSQL full-text search (BM25 via ts_rank_cd)
        Index(
            "idx_semantic_child_text_gin",
            text("to_tsvector('english', text_content)"),
            postgresql_using="gin",
        ),
    )


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class CanonicalEntity(Base):
    """Deduplicated, normalized form of an entity across all documents."""
    __tablename__ = "canonical_entity"

    canonical_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(Enum(EntityType), nullable=False)
    # Normalized value (lowercase, stripped, standardized)
    canonical_value = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    occurrences = relationship("ExtractedEntity", back_populates="canonical")

    __table_args__ = (
        # Composite B-Tree: fast lookup for (type, normalized_value) dedup checks
        Index("idx_canonical_entity_type_value", "entity_type", "canonical_value"),
    )


class ExtractedEntity(Base):
    __tablename__ = "extracted_entity"

    entity_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("semantic_child_chunk.chunk_id"), nullable=False)
    entity_type = Column(Enum(EntityType), nullable=False)
    entity_value = Column(String, nullable=False)
    # Extraction confidence (0–1); spaCy does not expose per-entity scores so
    # we use 1.0 for rule-based regex hits and GLiNER's native score otherwise
    confidence = Column(Float, nullable=True)
    # Which model/method produced this entity: "spacy", "gliner", "regex"
    extraction_method = Column(String, nullable=True)
    # FK to the deduplicated canonical form (populated async by canonicalization task)
    canonical_id = Column(UUID(as_uuid=True), ForeignKey("canonical_entity.canonical_id"), nullable=True)

    chunk = relationship("SemanticChildChunk", back_populates="entities")
    canonical = relationship("CanonicalEntity", back_populates="occurrences")

    __table_args__ = (
        # B-Tree (replaces old Hash): supports LIKE, range, and multi-column queries
        Index("idx_extracted_entity_value_btree", "entity_value", postgresql_using="btree"),
        # Composite index for (entity_type, entity_value) — used by deterministic bypass
        Index("idx_extracted_entity_type_value", "entity_type", "entity_value"),
    )


class KnowledgeGraphEdge(Base):
    __tablename__ = "knowledge_graph_edge"

    edge_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node = Column(UUID(as_uuid=True), index=True, nullable=False)
    target_node = Column(UUID(as_uuid=True), index=True, nullable=False)
    relationship_type = Column(Enum(RelationshipType), nullable=False)
    # Optional payload (e.g., contradiction details, address match score)
    edge_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


# ── Retrieval Audit ───────────────────────────────────────────────────────────

class RetrievalAuditLog(Base):
    __tablename__ = "retrieval_audit_log"

    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    router_decision = Column(String, nullable=False)
    confidence_scores = Column(JSONB)
    retrieved_chunks = Column(JSONB)
    # Per-stage wall-clock latencies in milliseconds:
    # {cache_check_ms, entity_extract_ms, hybrid_search_ms, reranker_ms,
    #  graph_traversal_ms, total_ms}
    stage_latencies = Column(JSONB)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

