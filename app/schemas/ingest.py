from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import UUID


class IngestResponse(BaseModel):
    doc_id: UUID
    status: str
    message: str
    task_id: str
    storage_uri: str
