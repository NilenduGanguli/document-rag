from celery import shared_task  # noqa: F401
from app.core.celery_app import celery_app
from typing import List, Dict, Any
from app.core.database import SessionLocal
from app.models.domain import RawDocument, ParsedLayoutSegment, SemanticChildChunk, ExtractedEntity, KnowledgeGraphEdge, RelationshipType, EntityType, FileType
from app.utils.file_utils import download_file
from app.services.nlp_service import extract_entities
from app.services.embedding_service import embedding_service
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import httpx
import uuid
import json
import os
import pandas as pd

@celery_app.task(bind=True, max_retries=3)
def process_kyc_document_task(self, doc_id: str, storage_uri: str):
    db = SessionLocal()
    try:
        # 1. Fetch raw document
        raw_doc = db.query(RawDocument).filter(RawDocument.doc_id == uuid.UUID(doc_id)).first()
        if not raw_doc:
            raise ValueError(f"Document {doc_id} not found")

        # Download file locally for processing
        local_path = download_file(storage_uri)

        layout_segments = []
        
        # 2. Layout Analysis
        if raw_doc.file_type in [FileType.PNG, FileType.JPEG]:
            # Call LLM OCR Service
            with open(local_path, "rb") as f:
                # Assuming ocr_service is running on port 8001 in docker network
                ocr_url = os.getenv("OCR_SERVICE_URL", "http://ocr_service:8001/ocr")
                response = httpx.post(ocr_url, files={"file": f}, timeout=120.0)
                response.raise_for_status()
                ocr_result = response.json()
                layout_segments.append({
                    "type": "Image", 
                    "content": ocr_result["text"],
                    "raw_text": ocr_result["text"]
                })
        elif raw_doc.file_type == FileType.XLSX:
            # Native Data (XLSX) parsed via native libraries directly into structured JSON
            df_dict = pd.read_excel(local_path, sheet_name=None)
            for sheet_name, df in df_dict.items():
                json_data = df.to_dict(orient="records")
                # Create a text representation for vector search
                text_repr = "\n".join([", ".join([f"{k}: {v}" for k, v in row.items()]) for row in json_data])
                layout_segments.append({
                    "type": "JSON", 
                    "content": json_data, 
                    "raw_text": text_repr,
                    "metadata": {"sheet": sheet_name}
                })
        else:
            # Use unstructured for PDF/DOCX
            if raw_doc.file_type == FileType.PDF:
                elements = partition_pdf(
                    filename=local_path,
                    strategy="hi_res", # Use high resolution strategy for OCR
                    infer_table_structure=True,
                    extract_images_in_pdf=True,
                    extract_image_block_types=["Image", "Table"]
                )
            else:
                elements = partition(filename=local_path)
                
            current_segment = []
            current_title = "Document Start"
            
            for el in elements:
                if el.category == "Title":
                    if current_segment:
                        layout_segments.append({
                            "type": "Text", 
                            "content": "\n".join(current_segment),
                            "raw_text": "\n".join(current_segment),
                            "metadata": {"section": current_title}
                        })
                        current_segment = []
                    current_title = str(el)
                elif el.category == "Table":
                    if current_segment:
                        layout_segments.append({
                            "type": "Text", 
                            "content": "\n".join(current_segment),
                            "raw_text": "\n".join(current_segment),
                            "metadata": {"section": current_title}
                        })
                        current_segment = []
                    
                    # Embedded tables explicitly extracted and converted to Markdown or HTML
                    html_table = el.metadata.text_as_html if hasattr(el, 'metadata') and hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html else str(el)
                    layout_segments.append({
                        "type": "Table",
                        "content": html_table,
                        "raw_text": str(el), # Raw text for child chunking
                        "metadata": {"section": current_title, "format": "html"}
                    })
                else:
                    current_segment.append(str(el))
                    
            if current_segment:
                layout_segments.append({
                    "type": "Text", 
                    "content": "\n".join(current_segment),
                    "raw_text": "\n".join(current_segment),
                    "metadata": {"section": current_title}
                })

        # Clean up local file
        if local_path != storage_uri and os.path.exists(local_path):
            os.remove(local_path)

        # 3. Semantic Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        child_chunks_data = []

        for segment_data in layout_segments:
            segment_id = uuid.uuid4()
            
            # Parent Node: represents broader document sections, tables, or raw image URIs
            segment = ParsedLayoutSegment(
                segment_id=segment_id,
                doc_id=raw_doc.doc_id,
                raw_content=segment_data
            )
            db.add(segment)
            
            # Child Nodes: highly granular, descriptive text optimized for vector search
            text_to_chunk = segment_data.get("raw_text", "")
            if not text_to_chunk:
                continue
                
            chunks = text_splitter.split_text(text_to_chunk)
            
            for chunk_text in chunks:
                chunk_id = uuid.uuid4()
                child_chunks_data.append({
                    "chunk_id": str(chunk_id),
                    "segment_id": str(segment_id),
                    "text": chunk_text
                })

        db.commit()
        
        # 4. Route to Batch Embedding
        if child_chunks_data:
            batch_embed_and_store.delay(child_chunks_data)

        return {"status": "success", "doc_id": doc_id}

    except Exception as exc:
        db.rollback()
        self.retry(exc=exc, countdown=60)
    finally:
        db.close()

from sqlalchemy.dialects.postgresql import insert

@celery_app.task
def batch_embed_and_store(child_chunks: List[Dict[str, Any]]):
    db = SessionLocal()
    try:
        # 1. Isolate text for batch embedding
        texts_to_embed = [chunk["text"] for chunk in child_chunks]
        
        # 2. Call the embedding model in one batch
        embeddings = embedding_service.get_embeddings(texts_to_embed)
        
        chunk_values = []
        entity_values = []
        edge_values = []

        _entity_type_map = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORG,
            "DATE": EntityType.DATE,
            "PASSPORT_NUM": EntityType.PASSPORT_NUM,
            "GPE": EntityType.ADDRESS,
            "ADDRESS": EntityType.ADDRESS,
        }

        for chunk_data, embedding in zip(child_chunks, embeddings):
            chunk_id = uuid.UUID(chunk_data["chunk_id"])
            segment_id = uuid.UUID(chunk_data["segment_id"])
            
            chunk_values.append({
                "chunk_id": chunk_id,
                "segment_id": segment_id,
                "text_content": chunk_data["text"],
                "dense_vector": embedding,
                "sparse_vector": {}
            })
            
            # 4. Entity Extraction (NER)
            entities = extract_entities(chunk_data["text"])
            for ent in entities:
                ent_type = _entity_type_map.get(ent["type"])
                if ent_type is None:
                    continue  # skip unrecognized entity types
                    
                # Deterministic UUID for idempotency
                ent_uuid = uuid.uuid5(chunk_id, f"{ent_type}_{ent['value']}")
                entity_values.append({
                    "entity_id": ent_uuid,
                    "chunk_id": chunk_id,
                    "entity_type": ent_type,
                    "entity_value": ent["value"]
                })
                
            # 5. Knowledge Graph Edges
            # Deterministic UUID for idempotency
            edge_uuid = uuid.uuid5(chunk_id, f"CHILD_OF_{segment_id}")
            edge_values.append({
                "edge_id": edge_uuid,
                "source_node": chunk_id,
                "target_node": segment_id,
                "relationship_type": RelationshipType.CHILD_OF
            })

        # 3. Bulk Insert into pgvector with ON CONFLICT DO UPDATE
        if chunk_values:
            stmt = insert(SemanticChildChunk).values(chunk_values)
            stmt = stmt.on_conflict_do_update(
                index_elements=['chunk_id'],
                set_={
                    'text_content': stmt.excluded.text_content,
                    'dense_vector': stmt.excluded.dense_vector
                }
            )
            db.execute(stmt)

        if entity_values:
            stmt_ent = insert(ExtractedEntity).values(entity_values)
            stmt_ent = stmt_ent.on_conflict_do_nothing(index_elements=['entity_id'])
            db.execute(stmt_ent)

        if edge_values:
            stmt_edge = insert(KnowledgeGraphEdge).values(edge_values)
            stmt_edge = stmt_edge.on_conflict_do_nothing(index_elements=['edge_id'])
            db.execute(stmt_edge)

        db.commit()
    except Exception as exc:
        db.rollback()
        raise exc
    finally:
        db.close()
