from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime


class ProcessingDirectives(BaseModel):
    # If True, use OCR even for PDF/DOCX (bypass layout extraction)
    force_ocr: bool = False
    # OCR backend: "gpt4v" | "azure_form_recognizer" | "tesseract"
    ocr_provider: str = "gpt4v"


class IngestResponse(BaseModel):
    doc_id: UUID
    status: str
    message: str
    task_id: str
    storage_uri: str


class EntityCount(BaseModel):
    entity_type: str
    count: int


class IngestStatusResponse(BaseModel):
    doc_id: UUID
    # "queued" | "processing" | "completed" | "failed"
    status: str
    layout_segments_created: Optional[int] = None
    semantic_chunks_created: Optional[int] = None
    entities_extracted: Optional[List[EntityCount]] = None
    graph_edges_created: Optional[int] = None
    processed_at: Optional[datetime] = None
    error: Optional[str] = None
