from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.ingest import EntityCount, IngestResponse, IngestStatusResponse, ProcessingDirectives
from app.models.domain import RawDocument, FileType
from app.worker.tasks import process_kyc_document_task
from app.core.celery_app import celery_app  # noqa: F401 — ensures Celery is configured in this process
from app.services.s3_service import upload_file
from typing import Optional
from uuid import UUID
import uuid
import os

router = APIRouter()

# Mapping of lowercase extensions → FileType enum values
_EXT_TO_FILE_TYPE: dict[str, str] = {
    ".pdf":  "PDF",
    ".png":  "PNG",
    ".jpg":  "JPEG",
    ".jpeg": "JPEG",
    ".tiff": "TIFF",
    ".tif":  "TIFF",
    ".docx": "DOCX",
    ".xlsx": "XLSX",
}

# Accepted MIME types → FileType enum values
_MIME_TO_FILE_TYPE: dict[str, str] = {
    "application/pdf":                                                          "PDF",
    "image/png":                                                                "PNG",
    "image/jpeg":                                                               "JPEG",
    "image/tiff":                                                               "TIFF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  "DOCX",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":        "XLSX",
    "application/msword":                                                        "DOCX",
    "application/vnd.ms-excel":                                                  "XLSX",
}


def _detect_mime(raw_bytes: bytes) -> str:
    """Use python-magic to detect MIME type from file bytes."""
    try:
        import magic
        return magic.from_buffer(raw_bytes, mime=True)
    except Exception:
        return ""


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    file: UploadFile = File(..., description="KYC document file (PDF, PNG, JPEG, TIFF, DOCX, XLSX)"),
    customer_id: str = Form(..., description="Unique customer identifier"),
    processing_directives_raw: Optional[str] = Form(
        None,
        description='Optional JSON processing directives: {"force_ocr": false, "ocr_provider": "gpt4v"}',
    ),
    db: Session = Depends(get_db),
):
    from app.core.config import settings

    # Parse processing directives (default to empty/defaults on missing/invalid JSON)
    try:
        directives = (
            ProcessingDirectives.model_validate_json(processing_directives_raw)
            if processing_directives_raw
            else ProcessingDirectives()
        )
    except Exception:
        directives = ProcessingDirectives()

    # Infer file type from extension
    _, ext = os.path.splitext((file.filename or "").lower())
    file_type_str = _EXT_TO_FILE_TYPE.get(ext)
    if file_type_str is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {list(_EXT_TO_FILE_TYPE)}",
        )

    # Read file bytes once — used for validation and upload
    raw_bytes = await file.read()

    # Empty file check
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # File size check
    if len(raw_bytes) > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {max_mb:.0f} MB.",
        )

    # MIME type validation — detects file type from actual bytes, not just extension
    detected_mime = _detect_mime(raw_bytes)
    if detected_mime and detected_mime not in _MIME_TO_FILE_TYPE:
        raise HTTPException(
            status_code=400,
            detail=f"Detected MIME type '{detected_mime}' is not accepted. "
                   f"Allowed types: {list(_MIME_TO_FILE_TYPE.keys())}",
        )

    # Cross-check: extension-declared type vs MIME-detected type
    if detected_mime and _MIME_TO_FILE_TYPE.get(detected_mime) != file_type_str:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' does not match detected content type '{detected_mime}'.",
        )

    file_type_enum = FileType(file_type_str)

    # Upload to MinIO
    doc_id = uuid.uuid4()
    object_key = f"{doc_id}/{file.filename}"
    storage_uri = upload_file(
        object_key=object_key,
        data=raw_bytes,
        content_type=file.content_type or "application/octet-stream",
    )

    # Persist document record (includes processing_directives for audit trail)
    raw_doc = RawDocument(
        doc_id=doc_id,
        customer_id=customer_id,
        file_type=file_type_enum,
        storage_uri=storage_uri,
        processing_directives=directives.model_dump(),
    )
    db.add(raw_doc)
    db.commit()
    db.refresh(raw_doc)

    # Dispatch Celery task with directives so OCR routing can be applied
    task = process_kyc_document_task.delay(str(doc_id), storage_uri, directives.model_dump())

    return IngestResponse(
        doc_id=doc_id,
        status="queued",
        message="Document queued for ingestion. Poll /ingest/{doc_id}/status for results.",
        task_id=task.id,
        storage_uri=storage_uri,
    )


@router.get("/ingest/{doc_id}/status", response_model=IngestStatusResponse)
def get_ingest_status(doc_id: UUID, db: Session = Depends(get_db)):
    """Poll the processing status and pipeline counts for an ingested document."""
    raw_doc = db.query(RawDocument).filter(RawDocument.doc_id == doc_id).first()
    if not raw_doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    summary = raw_doc.processing_summary or {}

    if raw_doc.processed_at:
        doc_status = "completed"
    elif summary.get("error"):
        doc_status = "failed"
    else:
        doc_status = "queued"

    # Build entity count list from summary dict {PERSON: 3, ORG: 1, ...}
    entity_counts = None
    if summary.get("entities"):
        entity_counts = [
            EntityCount(entity_type=k, count=v)
            for k, v in summary["entities"].items()
        ]

    return IngestStatusResponse(
        doc_id=raw_doc.doc_id,
        status=doc_status,
        layout_segments_created=summary.get("segments"),
        semantic_chunks_created=summary.get("chunks"),
        entities_extracted=entity_counts,
        graph_edges_created=summary.get("edges"),
        processed_at=raw_doc.processed_at,
        error=summary.get("error"),
    )
