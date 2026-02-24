from pydantic_settings import BaseSettings
from typing import Optional, Set

class Settings(BaseSettings):
    PROJECT_NAME: str = "KYC Document RAG API"
    API_V1_STR: str = "/api/v1"

    # ── Database ──────────────────────────────────────────────────────────────
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "kyc_rag"
    POSTGRES_PORT: str = "5432"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"

    @property
    def CELERY_BROKER_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/1"

    # ── MinIO / S3 ────────────────────────────────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "kyc-documents"
    MINIO_SECURE: bool = False

    # ── Embeddings & Reranking ────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    # Sparse embedding model (SPLADE-based via fastembed)
    SPARSE_EMBEDDING_MODEL_NAME: str = "prithivida/Splade_PP_en_v1"
    # Maximum texts per embedding batch (prevents OOM on large documents)
    EMBEDDING_BATCH_SIZE: int = 256
    # Redis TTL (seconds) for query-embedding cache
    QUERY_EMBEDDING_CACHE_TTL: int = 60
    # Timeout (seconds) for cross-encoder reranker predict(); fallback to RRF on timeout
    RERANKER_TIMEOUT_S: float = 3.0

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Maximum file size accepted at the ingest endpoint (bytes). Default 50 MB.
    MAX_FILE_SIZE_BYTES: int = 52_428_800
    # Target chunk size in characters for text segments
    CHUNK_SIZE: int = 500
    # Character overlap between adjacent text chunks
    CHUNK_OVERLAP: int = 50
    # Minimum acceptable chunk length in characters; shorter chunks are dropped
    MIN_CHUNK_LENGTH: int = 50
    # Maximum Excel rows per chunk group
    EXCEL_ROWS_PER_CHUNK: int = 20
    # OCR service endpoint (override for local / alternative backends)
    OCR_SERVICE_URL: str = "http://ocr_service:8001/ocr"
    # Number of OCR retry attempts before falling back to Tesseract
    OCR_RETRY_ATTEMPTS: int = 3
    # Wait seconds between OCR retry attempts
    OCR_RETRY_WAIT_S: float = 2.0

    # ── Retrieval / Query Router ──────────────────────────────────────────────
    # Entity types that trigger the deterministic exact-match bypass.
    # Stored as a comma-separated string so it can be overridden via env var.
    DETERMINISTIC_BYPASS_ENTITY_TYPES: str = "PASSPORT_NUM,ORG,ADDRESS"

    @property
    def bypass_entity_types(self) -> Set[str]:
        return {t.strip() for t in self.DETERMINISTIC_BYPASS_ENTITY_TYPES.split(",") if t.strip()}

    # RRF smoothing constant k — standard default is 60
    RRF_K: int = 60
    # Weight given to the dense (semantic) search leg in RRF (0–1).
    # Sparse weight = 1 - RRF_DENSE_WEIGHT.
    RRF_DENSE_WEIGHT: float = 0.7
    # Cross-encoder confidence cutoff; chunks below this score are discarded
    RERANKER_CONFIDENCE_CUTOFF: float = 0.5
    # Minimum child-chunk count per parent before parent block is returned instead
    GRAPH_FREQUENCY_THRESHOLD: int = 2
    # Reranker score above which parent block is also returned (semantic threshold)
    GRAPH_SEMANTIC_THRESHOLD: float = 0.8
    # Maximum hops in multi-hop graph traversal (cap enforced in SQL)
    GRAPH_MAX_DEPTH: int = 2
    # Redis TTL (seconds) for query-result semantic cache
    QUERY_CACHE_TTL: int = 3600

    # ── Azure OCR (optional — activate by setting AZURE_FORM_RECOGNIZER_ENDPOINT) ─
    AZURE_FORM_RECOGNIZER_ENDPOINT: Optional[str] = None
    AZURE_FORM_RECOGNIZER_KEY: Optional[str] = None
    # Default OCR backend: "gpt4v" | "azure_form_recognizer" | "tesseract"
    DEFAULT_OCR_PROVIDER: str = "gpt4v"
    # Azure Document Intelligence model ID (prebuilt-read = text-only OCR)
    AZURE_FORM_RECOGNIZER_MODEL_ID: str = "prebuilt-read"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
