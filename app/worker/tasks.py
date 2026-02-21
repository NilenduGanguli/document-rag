from celery import shared_task
from typing import List, Dict, Any
from app.core.database import SessionLocal
from app.models.domain import RawDocument, ParsedLayoutSegment, SemanticChildChunk, ExtractedEntity, KnowledgeGraphEdge, RelationshipType, EntityType
import uuid
import json

@shared_task(bind=True, max_retries=3)
def process_kyc_document_task(self, doc_id: str, storage_uri: str):
    db = SessionLocal()
    try:
        # 1. Fetch raw document
        raw_doc = db.query(RawDocument).filter(RawDocument.doc_id == uuid.UUID(doc_id)).first()
        if not raw_doc:
            raise ValueError(f"Document {doc_id} not found")

        # 2. Layout Analysis (Mock Azure OCR / unstructured.io)
        # In a real scenario, this would call an OCR service
        segment_id = uuid.uuid4()
        raw_content = {"text": "Mock extracted text from OCR", "tables": []}
        
        segment = ParsedLayoutSegment(
            segment_id=segment_id,
            doc_id=raw_doc.doc_id,
            raw_content=raw_content
        )
        db.add(segment)
        db.commit()

        # 3. Semantic Chunking (Mock)
        # In a real scenario, this would split the text into chunks
        chunks = [
            {"text": "Mock chunk 1", "vector": [0.1] * 1024},
            {"text": "Mock chunk 2", "vector": [0.2] * 1024}
        ]
        
        for chunk_data in chunks:
            chunk_id = uuid.uuid4()
            chunk = SemanticChildChunk(
                chunk_id=chunk_id,
                segment_id=segment_id,
                text_content=chunk_data["text"],
                dense_vector=chunk_data["vector"],
                sparse_vector={"mock": "sparse"}
            )
            db.add(chunk)
            
            # 4. Entity Extraction (Mock NER)
            # In a real scenario, this would call an NLP service
            if "passport" in chunk_data["text"].lower():
                entity = ExtractedEntity(
                    entity_id=uuid.uuid4(),
                    chunk_id=chunk_id,
                    entity_type=EntityType.PASSPORT_NUM,
                    entity_value="A1234"
                )
                db.add(entity)
                
            # 5. Knowledge Graph Edges (Mock)
            edge = KnowledgeGraphEdge(
                edge_id=uuid.uuid4(),
                source_node=chunk_id,
                target_node=segment_id,
                relationship_type=RelationshipType.CHILD_OF
            )
            db.add(edge)

        db.commit()
        return {"status": "success", "doc_id": doc_id}

    except Exception as exc:
        db.rollback()
        self.retry(exc=exc, countdown=60)
    finally:
        db.close()

@shared_task
def batch_embed_and_store(child_chunks: List[Dict[str, Any]]):
    # Mock batch embedding
    # In a real scenario, this would call an embedding service
    db = SessionLocal()
    try:
        for chunk_data in child_chunks:
            chunk = SemanticChildChunk(
                chunk_id=uuid.UUID(chunk_data["chunk_id"]),
                segment_id=uuid.UUID(chunk_data["segment_id"]),
                text_content=chunk_data["text"],
                dense_vector=[0.1] * 1024, # Mock vector
                sparse_vector={"mock": "sparse"}
            )
            db.add(chunk)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise exc
    finally:
        db.close()
