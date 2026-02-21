import uuid
from sqlalchemy import Column, String, Text, Enum, ForeignKey, JSON, Float, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base
import enum

class FileType(str, enum.Enum):
    PDF = "PDF"
    PNG = "PNG"
    JPEG = "JPEG"
    DOCX = "DOCX"
    XLSX = "XLSX"

class EntityType(str, enum.Enum):
    PERSON = "PERSON"
    PASSPORT_NUM = "PASSPORT_NUM"
    ORG = "ORG"
    DATE = "DATE"
    ADDRESS = "ADDRESS"

class RelationshipType(str, enum.Enum):
    CHILD_OF = "CHILD_OF"
    SAME_ADDRESS = "SAME_ADDRESS"
    CONTRADICTS = "CONTRADICTS"
    REFERENCES_UBO = "REFERENCES_UBO"

class RawDocument(Base):
    __tablename__ = "raw_document"
    
    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(String, index=True, nullable=False)
    file_type = Column(Enum(FileType), nullable=False)
    storage_uri = Column(String, nullable=False)
    
    segments = relationship("ParsedLayoutSegment", back_populates="document")

class ParsedLayoutSegment(Base):
    __tablename__ = "parsed_layout_segment"
    
    segment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("raw_document.doc_id"), nullable=False)
    raw_content = Column(JSONB, nullable=False)
    
    document = relationship("RawDocument", back_populates="segments")
    chunks = relationship("SemanticChildChunk", back_populates="segment")

class SemanticChildChunk(Base):
    __tablename__ = "semantic_child_chunk"
    
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    segment_id = Column(UUID(as_uuid=True), ForeignKey("parsed_layout_segment.segment_id"), nullable=False)
    text_content = Column(Text, nullable=False) # GIN index for BM25
    dense_vector = Column(Vector(1024)) # HNSW index
    sparse_vector = Column(JSONB)
    
    segment = relationship("ParsedLayoutSegment", back_populates="chunks")
    entities = relationship("ExtractedEntity", back_populates="chunk")

class ExtractedEntity(Base):
    __tablename__ = "extracted_entity"
    
    entity_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("semantic_child_chunk.chunk_id"), nullable=False)
    entity_type = Column(Enum(EntityType), nullable=False)
    entity_value = Column(String, index=True, nullable=False) # Hash index
    
    chunk = relationship("SemanticChildChunk", back_populates="entities")

class KnowledgeGraphEdge(Base):
    __tablename__ = "knowledge_graph_edge"
    
    edge_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node = Column(UUID(as_uuid=True), index=True, nullable=False)
    target_node = Column(UUID(as_uuid=True), index=True, nullable=False)
    relationship_type = Column(Enum(RelationshipType), nullable=False)

class RetrievalAuditLog(Base):
    __tablename__ = "retrieval_audit_log"
    
    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    router_decision = Column(String, nullable=False)
    confidence_scores = Column(JSONB)
    retrieved_chunks = Column(JSONB)
