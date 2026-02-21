from fastapi import FastAPI
from app.api.v1 import ingest, retrieve
from app.core.config import settings
from app.core.database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(ingest.router, prefix=settings.API_V1_STR, tags=["ingestion"])
app.include_router(retrieve.router, prefix=settings.API_V1_STR, tags=["retrieval"])

@app.get("/")
def root():
    return {"message": "Welcome to the KYC Document RAG API"}
