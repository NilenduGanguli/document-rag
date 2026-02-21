from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.ingest import IngestRequest, IngestResponse
from app.models.domain import RawDocument, FileType
from app.worker.tasks import process_kyc_document_task
import uuid

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_document(request: IngestRequest, db: Session = Depends(get_db)):
    try:
        file_type_enum = FileType(request.file_type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {request.file_type}")

    # Create raw document record
    doc_id = uuid.uuid4()
    raw_doc = RawDocument(
        doc_id=doc_id,
        customer_id=request.customer_id,
        file_type=file_type_enum,
        storage_uri=str(request.storage_uri)
    )
    db.add(raw_doc)
    db.commit()
    db.refresh(raw_doc)

    # Dispatch Celery task
    task = process_kyc_document_task.delay(str(doc_id), str(request.storage_uri))

    return IngestResponse(
        doc_id=doc_id,
        status="processing",
        message="Document ingestion started asynchronously.",
        task_id=task.id
    )
