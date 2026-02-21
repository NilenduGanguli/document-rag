from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from uuid import UUID

class IngestRequest(BaseModel):
    customer_id: str
    file_type: str
    storage_uri: HttpUrl
    metadata: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    doc_id: UUID
    status: str
    message: str
    task_id: str
