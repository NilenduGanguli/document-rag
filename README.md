# KYC Document RAG API

This is a production-ready Retrieval-Augmented Generation (RAG) system tailored for Know Your Customer (KYC) compliance.

## Architecture

*   **API**: FastAPI
*   **Background Tasks**: Celery + Redis
*   **Database**: PostgreSQL + pgvector
*   **ORM**: SQLAlchemy

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```

## Endpoints

*   `POST /api/v1/ingest`: Ingest a KYC document asynchronously.
*   `POST /api/v1/retrieve`: Retrieve context for a query using Hybrid Search and Deterministic Bypass.

## Design

See `kyc_document_rag_design.md` for the detailed architecture and design philosophy.