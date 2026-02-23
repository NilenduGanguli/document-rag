from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.ingest import IngestResponse
from app.models.domain import RawDocument, FileType
from app.worker.tasks import process_kyc_document_task
from app.core.celery_app import celery_app  # noqa: F401 — ensures Celery is configured in this process
from app.services.s3_service import upload_file
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
}


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    file: UploadFile = File(..., description="KYC document file (PDF, PNG, JPEG, TIFF)"),
    customer_id: str = Form(..., description="Unique customer identifier"),
    db: Session = Depends(get_db),
):
    # Infer file type from extension
    _, ext = os.path.splitext((file.filename or "").lower())
    file_type_str = _EXT_TO_FILE_TYPE.get(ext)
    if file_type_str is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {list(_EXT_TO_FILE_TYPE)}",
        )
    file_type_enum = FileType(file_type_str)

    # Upload to MinIO
    doc_id = uuid.uuid4()
    object_key = f"{doc_id}/{file.filename}"
    raw_bytes = await file.read()
    storage_uri = upload_file(
        object_key=object_key,
        data=raw_bytes,
        content_type=file.content_type or "application/octet-stream",
    )

    # Persist document record
    raw_doc = RawDocument(
        doc_id=doc_id,
        customer_id=customer_id,
        file_type=file_type_enum,
        storage_uri=storage_uri,
    )
    db.add(raw_doc)
    db.commit()
    db.refresh(raw_doc)

    # Dispatch Celery task
    task = process_kyc_document_task.delay(str(doc_id), storage_uri)

    return IngestResponse(
        doc_id=doc_id,
        status="processing",
        message="Document ingestion started asynchronously.",
        task_id=task.id,
        storage_uri=storage_uri,
    )
