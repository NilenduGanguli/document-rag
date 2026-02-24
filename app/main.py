from fastapi import FastAPI
from sqlalchemy import text
from contextlib import asynccontextmanager
import asyncio

from app.api.v1 import ingest, retrieve
from app.core.config import settings
from app.core.database import engine, Base
from app.core.migrations import run_startup_migrations
from app.services.embedding_service import embedding_service

# Create pgvector extension before creating tables
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

# Create any tables that don't exist yet (fresh database)
Base.metadata.create_all(bind=engine)

# Apply incremental DDL (new columns / enum values) to pre-existing tables.
# This is idempotent and safe to run on every startup.
run_startup_migrations(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the embedding model in a thread executor before accepting requests.
    # This ensures BAAI/bge-m3 is warm when the first retrieval request arrives.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: embedding_service.model)
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

app.include_router(ingest.router, prefix=settings.API_V1_STR, tags=["ingestion"])
app.include_router(retrieve.router, prefix=settings.API_V1_STR, tags=["retrieval"])

@app.get("/")
def root():
    return {"message": "Welcome to the KYC Document RAG API"}
