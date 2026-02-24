"""
Celery tasks for KYC document ingestion.

Pipeline:
  1. process_kyc_document_task  — layout analysis, content-type-aware chunking
  2. batch_embed_and_store       — dense+sparse embedding, NER with windowed context,
                                   knowledge graph edge construction, cache invalidation
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from celery import shared_task  # noqa: F401
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.dialects.postgresql import insert
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.domain import (
    EntityType,
    ExtractedEntity,
    FileType,
    KnowledgeGraphEdge,
    ParsedLayoutSegment,
    RawDocument,
    RelationshipType,
    SemanticChildChunk,
)
from app.services.embedding_service import embedding_service
from app.services.nlp_service import extract_entities_windowed
from app.utils.file_utils import download_file

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Entity type mapping
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_TYPE_MAP: Dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORG,
    "DATE": EntityType.DATE,
    "GPE": EntityType.ADDRESS,
    "ADDRESS": EntityType.ADDRESS,
    "PASSPORT_NUM": EntityType.PASSPORT_NUM,
    "UBO": EntityType.UBO,
    "BENEFICIAL_OWNER": EntityType.BENEFICIAL_OWNER,
    "COMPANY_REG_NUM": EntityType.COMPANY_REG_NUM,
    "TAX_ID": EntityType.TAX_ID,
    "IBAN": EntityType.IBAN,
    "SANCTION_ENTRY": EntityType.SANCTION_ENTRY,
}

# ─────────────────────────────────────────────────────────────────────────────
# HTML table → structured JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

class _TableHTMLParser(HTMLParser):
    """Minimal HTML parser that extracts rows and cells from a table string."""

    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: List[str] = []
        self._cell_buf: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("tr",):
            self._row = []
        elif tag in ("td", "th"):
            self._cell_buf = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._row.append(" ".join(self._cell_buf).strip())
            self._in_cell = False
        elif tag == "tr":
            if self._row:
                self.rows.append(self._row[:])

    def handle_data(self, data):
        if self._in_cell:
            stripped = data.strip()
            if stripped:
                self._cell_buf.append(stripped)


def _html_table_to_json(html: str) -> Dict[str, Any]:
    """Convert an HTML table string to {"headers": [...], "rows": [[...]]}."""
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return {"headers": [], "rows": []}

    all_rows = parser.rows
    if not all_rows:
        return {"headers": [], "rows": []}

    headers = all_rows[0]
    data_rows = all_rows[1:]
    return {"headers": headers, "rows": data_rows}


def _table_json_to_text(table_json: Dict[str, Any]) -> str:
    """Convert structured table JSON to embeddable text that preserves column context."""
    headers = table_json.get("headers", [])
    rows = table_json.get("rows", [])
    lines: List[str] = []
    if headers:
        lines.append(" | ".join(headers))
        lines.append("—" * 40)
    for row in rows:
        if headers:
            parts = [f"{h}: {v}" for h, v in zip(headers, row) if v]
            lines.append(" | ".join(parts))
        else:
            lines.append(" | ".join(row))
    return "\n".join(lines)


def _table_json_to_row_chunks(
    table_json: Dict[str, Any], rows_per_chunk: int = 20
) -> List[str]:
    """Split a large table into text chunks, each prefixed with column headers."""
    headers = table_json.get("headers", [])
    rows = table_json.get("rows", [])
    header_line = " | ".join(headers) if headers else ""
    chunks = []
    for i in range(0, len(rows), rows_per_chunk):
        batch = rows[i : i + rows_per_chunk]
        lines: List[str] = []
        if header_line:
            lines.append(header_line)
        for row in batch:
            if headers:
                parts = [f"{h}: {v}" for h, v in zip(headers, row) if v]
                lines.append(" | ".join(parts))
            else:
                lines.append(" | ".join(row))
        chunks.append("\n".join(lines))
    return chunks if chunks else [""]


# ─────────────────────────────────────────────────────────────────────────────
# OCR helpers — three backends + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(settings.OCR_RETRY_ATTEMPTS),
    wait=wait_fixed(settings.OCR_RETRY_WAIT_S),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _call_ocr_service(local_path: str) -> str:
    """Call the LLM OCR microservice; retries on transient HTTP/timeout errors."""
    with open(local_path, "rb") as fh:
        response = httpx.post(
            settings.OCR_SERVICE_URL,
            files={"file": fh},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json().get("text", "")


def _ocr_gpt4v(local_path: str) -> str:
    """OCR via the GPT-4o-mini microservice (primary LLM-based OCR)."""
    return _call_ocr_service(local_path)


def _ocr_tesseract(local_path: str) -> str:
    """OCR via local Tesseract (requires tesseract-ocr system package + pytesseract)."""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(local_path))
    except Exception as exc:
        logger.error("Tesseract OCR failed: %s. Returning empty string.", exc)
        return ""


def _ocr_azure(local_path: str) -> str:
    """
    OCR via Azure Document Intelligence (Form Recognizer).
    Requires AZURE_FORM_RECOGNIZER_ENDPOINT and AZURE_FORM_RECOGNIZER_KEY in settings.
    """
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential

        client = DocumentAnalysisClient(
            endpoint=settings.AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_FORM_RECOGNIZER_KEY),
        )
        with open(local_path, "rb") as f:
            poller = client.begin_analyze_document(
                settings.AZURE_FORM_RECOGNIZER_MODEL_ID, f
            )
        result = poller.result()
        return "\n".join(p.content for p in result.paragraphs) if result.paragraphs else ""
    except Exception as exc:
        logger.error("Azure Form Recognizer OCR failed: %s. Returning empty string.", exc)
        return ""


def _ocr_image(local_path: str, provider: Optional[str] = None) -> str:
    """
    Dispatch OCR to the appropriate backend based on `provider`.

    Provider routing:
      "azure_form_recognizer"  → Azure Document Intelligence (if endpoint is configured)
      "tesseract"              → Local Tesseract
      "gpt4v" or None          → GPT-4o-mini microservice (fallback: Tesseract on error)
    """
    provider = provider or settings.DEFAULT_OCR_PROVIDER

    if provider == "azure_form_recognizer" and settings.AZURE_FORM_RECOGNIZER_ENDPOINT:
        return _ocr_azure(local_path)

    if provider == "tesseract":
        return _ocr_tesseract(local_path)

    # Default: GPT-4o-mini microservice with Tesseract fallback
    try:
        return _ocr_gpt4v(local_path)
    except Exception as primary_exc:
        logger.warning(
            "OCR service failed for %s (%s). Attempting Tesseract fallback.",
            local_path,
            primary_exc,
        )
        return _ocr_tesseract(local_path)


# ─────────────────────────────────────────────────────────────────────────────
# DOCX partition (unstructured)
# ─────────────────────────────────────────────────────────────────────────────

def _partition_docx(local_path: str):
    """Use unstructured partition_docx if available, else fall back to partition."""
    try:
        from unstructured.partition.docx import partition_docx
        return partition_docx(filename=local_path)
    except ImportError:
        return partition(filename=local_path)


# ─────────────────────────────────────────────────────────────────────────────
# Content-type-aware chunking helpers
# ─────────────────────────────────────────────────────────────────────────────

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)

_MAX_TABLE_SINGLE_CHUNK_CHARS = settings.CHUNK_SIZE * 6  # ~3000 chars


def _chunk_segment(segment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of chunk dicts for a single segment.
    Each dict: {text, embed_text, content_type, position_in_segment, has_table}

    Strategy:
      Table  → one chunk per table (or row-groups if large), header preserved per chunk
      Image  → one chunk (OCR text is already a single block)
      JSON   → one chunk per row-group (EXCEL_ROWS_PER_CHUNK rows), header always included
      Text   → RecursiveCharacterTextSplitter
    """
    seg_type = segment_data.get("type", "Text")
    section = segment_data.get("metadata", {}).get("section", "")
    raw_text = segment_data.get("raw_text", "")

    chunks: List[Dict[str, Any]] = []

    def _prefix(ctype: str) -> str:
        parts = []
        if section:
            parts.append(f"Section: {section}")
        parts.append(f"Type: {ctype}")
        return "[" + " | ".join(parts) + "] "

    def _make(text: str, embed_text: str, pos: int, has_table: bool) -> Dict[str, Any]:
        return {
            "text": text,
            "embed_text": embed_text,
            "content_type": seg_type,
            "position_in_segment": pos,
            "has_table": has_table,
        }

    if seg_type == "Table":
        table_json = segment_data.get("content", {})
        if isinstance(table_json, dict) and table_json.get("headers") is not None:
            table_text = _table_json_to_text(table_json)
        else:
            # Fallback: use raw_text
            table_text = raw_text

        if len(table_text) <= _MAX_TABLE_SINGLE_CHUNK_CHARS:
            if len(table_text.strip()) >= settings.MIN_CHUNK_LENGTH:
                chunks.append(_make(
                    table_text,
                    _prefix("Table") + table_text,
                    0, True,
                ))
        else:
            sub_chunks = _table_json_to_row_chunks(
                table_json if isinstance(table_json, dict) else {},
                rows_per_chunk=settings.EXCEL_ROWS_PER_CHUNK,
            )
            for pos, sub in enumerate(sub_chunks):
                if len(sub.strip()) >= settings.MIN_CHUNK_LENGTH:
                    chunks.append(_make(
                        sub, _prefix("Table") + sub, pos, True,
                    ))

    elif seg_type == "Image":
        if len(raw_text.strip()) >= settings.MIN_CHUNK_LENGTH:
            chunks.append(_make(raw_text, _prefix("Image") + raw_text, 0, False))

    elif seg_type == "JSON":
        # Excel row-group chunks
        rows = segment_data.get("content", [])
        headers = list(rows[0].keys()) if rows else []
        header_line = " | ".join(headers)

        for i in range(0, max(1, len(rows)), settings.EXCEL_ROWS_PER_CHUNK):
            batch = rows[i : i + settings.EXCEL_ROWS_PER_CHUNK]
            lines = [header_line] if header_line else []
            for row in batch:
                lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
            chunk_text = "\n".join(lines)
            if len(chunk_text.strip()) >= settings.MIN_CHUNK_LENGTH:
                chunks.append(_make(
                    chunk_text, _prefix("Spreadsheet") + chunk_text,
                    i // settings.EXCEL_ROWS_PER_CHUNK, False,
                ))

    else:  # Text
        if not raw_text.strip():
            return chunks
        sub_texts = _text_splitter.split_text(raw_text)
        for pos, sub in enumerate(sub_texts):
            if len(sub.strip()) >= settings.MIN_CHUNK_LENGTH:
                chunks.append(_make(
                    sub, _prefix("Text") + sub, pos, False,
                ))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Layout analysis + chunking
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3)
def process_kyc_document_task(self, doc_id: str, storage_uri: str, directives: Optional[dict] = None):
    directives = directives or {}
    force_ocr = directives.get("force_ocr", False)
    ocr_provider = directives.get("ocr_provider", settings.DEFAULT_OCR_PROVIDER)

    db = SessionLocal()
    try:
        raw_doc = db.query(RawDocument).filter(
            RawDocument.doc_id == uuid.UUID(doc_id)
        ).first()
        if not raw_doc:
            raise ValueError(f"Document {doc_id} not found")

        local_path = download_file(storage_uri)
        layout_segments: List[Dict[str, Any]] = []

        # ── Layout analysis ───────────────────────────────────────────────────

        if raw_doc.file_type in (FileType.PNG, FileType.JPEG):
            text = _ocr_image(local_path, ocr_provider)
            layout_segments.append({
                "type": "Image",
                "content": text,
                "raw_text": text,
                "metadata": {},
            })

        elif raw_doc.file_type == FileType.XLSX:
            df_dict = pd.read_excel(local_path, sheet_name=None)
            for sheet_name, df in df_dict.items():
                # Preserve column dtypes as strings for the schema note
                dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
                records = df.fillna("").to_dict(orient="records")
                layout_segments.append({
                    "type": "JSON",
                    "content": records,
                    "raw_text": "",   # row-level text built in _chunk_segment
                    "metadata": {"sheet": sheet_name, "dtypes": dtypes},
                })

        elif force_ocr and raw_doc.file_type in (FileType.PDF, FileType.DOCX):
            # force_ocr=True: bypass layout extraction and use OCR directly
            text = _ocr_image(local_path, ocr_provider)
            layout_segments.append({
                "type": "Image",
                "content": text,
                "raw_text": text,
                "metadata": {"force_ocr": True, "ocr_provider": ocr_provider},
            })

        else:
            # PDF → unstructured hi_res; DOCX → partition_docx
            if raw_doc.file_type == FileType.PDF:
                elements = partition_pdf(
                    filename=local_path,
                    strategy="hi_res",
                    infer_table_structure=True,
                    extract_images_in_pdf=True,
                    extract_image_block_types=["Image", "Table"],
                )
            else:
                elements = _partition_docx(local_path)

            current_texts: List[str] = []
            current_title = "Document Start"

            def _flush_text():
                nonlocal current_texts, current_title
                if current_texts:
                    body = "\n".join(current_texts)
                    layout_segments.append({
                        "type": "Text",
                        "content": body,
                        "raw_text": body,
                        "metadata": {"section": current_title},
                    })
                    current_texts = []

            for el in elements:
                if el.category == "Title":
                    _flush_text()
                    current_title = str(el)
                elif el.category == "Table":
                    _flush_text()
                    html = (
                        el.metadata.text_as_html
                        if hasattr(el, "metadata")
                        and hasattr(el.metadata, "text_as_html")
                        and el.metadata.text_as_html
                        else ""
                    )
                    # Convert HTML → structured JSON (rows + headers)
                    table_json = _html_table_to_json(html) if html else {"headers": [], "rows": []}
                    raw_text = _table_json_to_text(table_json) or str(el)
                    layout_segments.append({
                        "type": "Table",
                        "content": table_json,   # structured, not raw HTML
                        "raw_text": raw_text,
                        "metadata": {"section": current_title},
                    })
                else:
                    current_texts.append(str(el))

            _flush_text()

        # Clean up temp file
        if local_path != storage_uri and os.path.exists(local_path):
            os.remove(local_path)

        # ── Content-type-aware chunking ───────────────────────────────────────

        child_chunks_data: List[Dict[str, Any]] = []

        for segment_data in layout_segments:
            segment_id = uuid.uuid4()

            segment = ParsedLayoutSegment(
                segment_id=segment_id,
                doc_id=raw_doc.doc_id,
                raw_content=segment_data,
            )
            db.add(segment)

            chunk_dicts = _chunk_segment(segment_data)
            for pos, cd in enumerate(chunk_dicts):
                child_chunks_data.append({
                    "chunk_id": str(uuid.uuid4()),
                    "segment_id": str(segment_id),
                    "text": cd["text"],
                    "embed_text": cd["embed_text"],
                    "chunk_metadata": {
                        "char_count": len(cd["text"]),
                        "content_type": cd["content_type"],
                        "position_in_segment": cd["position_in_segment"],
                        "has_table": cd["has_table"],
                        "embed_prefix": cd["embed_text"][: cd["embed_text"].index("]") + 2]
                        if "]" in cd["embed_text"]
                        else "",
                    },
                })

        db.commit()

        if child_chunks_data:
            batch_embed_and_store.delay(child_chunks_data, str(raw_doc.customer_id), doc_id)

        return {"status": "success", "doc_id": doc_id}

    except Exception as exc:
        db.rollback()
        # Write error summary so status endpoint reports "failed"
        try:
            db.query(RawDocument).filter(
                RawDocument.doc_id == uuid.UUID(doc_id)
            ).update({"processing_summary": {"error": str(exc)}})
            db.commit()
        except Exception:
            pass
        self.retry(exc=exc, countdown=60)
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge graph edge builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_same_address_edges(
    entity_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Group ADDRESS entities by normalized value.
    For each address shared by > 1 chunk, create SAME_ADDRESS edges between
    all unique chunk pairs.
    """
    address_to_chunks: Dict[str, List[uuid.UUID]] = defaultdict(list)
    for ent in entity_rows:
        if ent["entity_type"] == EntityType.ADDRESS:
            normalized = ent["entity_value"].lower().strip()
            chunk_id = ent["chunk_id"]
            if chunk_id not in address_to_chunks[normalized]:
                address_to_chunks[normalized].append(chunk_id)

    edges = []
    for addr, chunk_ids in address_to_chunks.items():
        # Create edges between each pair
        for i in range(len(chunk_ids)):
            for j in range(i + 1, len(chunk_ids)):
                edge_id = uuid.uuid5(
                    chunk_ids[i], f"SAME_ADDRESS_{addr}_{chunk_ids[j]}"
                )
                edges.append({
                    "edge_id": edge_id,
                    "source_node": chunk_ids[i],
                    "target_node": chunk_ids[j],
                    "relationship_type": RelationshipType.SAME_ADDRESS,
                    "metadata": {"normalized_address": addr},
                })
    return edges


def _build_contradicts_edges(
    entity_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect DATE contradictions involving the same PERSON/ORG.

    Strategy: for each PERSON/ORG value, collect all (chunk_id, DATE-values)
    pairs. If a person appears in ≥ 2 chunks that have different DATE entities,
    create a CONTRADICTS edge between those chunks.
    """
    # chunk_id → set of DATE values
    chunk_dates: Dict[uuid.UUID, set] = defaultdict(set)
    for ent in entity_rows:
        if ent["entity_type"] == EntityType.DATE:
            chunk_dates[ent["chunk_id"]].add(ent["entity_value"])

    # person_value → list of chunk_ids containing that person
    person_chunks: Dict[str, List[uuid.UUID]] = defaultdict(list)
    for ent in entity_rows:
        if ent["entity_type"] in (EntityType.PERSON, EntityType.ORG):
            val = ent["entity_value"].lower().strip()
            if ent["chunk_id"] not in person_chunks[val]:
                person_chunks[val].append(ent["chunk_id"])

    edges = []
    for person_val, chunk_ids in person_chunks.items():
        if len(chunk_ids) < 2:
            continue
        for i in range(len(chunk_ids)):
            for j in range(i + 1, len(chunk_ids)):
                dates_i = chunk_dates.get(chunk_ids[i], set())
                dates_j = chunk_dates.get(chunk_ids[j], set())
                if dates_i and dates_j and dates_i != dates_j:
                    edge_id = uuid.uuid5(
                        chunk_ids[i], f"CONTRADICTS_{person_val}_{chunk_ids[j]}"
                    )
                    edges.append({
                        "edge_id": edge_id,
                        "source_node": chunk_ids[i],
                        "target_node": chunk_ids[j],
                        "relationship_type": RelationshipType.CONTRADICTS,
                        "metadata": {
                            "entity": person_val,
                            "dates_a": list(dates_i),
                            "dates_b": list(dates_j),
                        },
                    })
    return edges


def _build_ubo_edges(
    entity_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each UBO / BENEFICIAL_OWNER entity, find ORG entities in other chunks
    of the same document and create REFERENCES_UBO edges (ORG → UBO).
    """
    ubo_chunks = [
        ent for ent in entity_rows
        if ent["entity_type"] in (EntityType.UBO, EntityType.BENEFICIAL_OWNER)
    ]
    org_chunks = [
        ent for ent in entity_rows
        if ent["entity_type"] == EntityType.ORG
    ]

    edges = []
    for ubo_ent in ubo_chunks:
        for org_ent in org_chunks:
            if org_ent["chunk_id"] == ubo_ent["chunk_id"]:
                continue  # Skip self-referential edges
            edge_id = uuid.uuid5(
                org_ent["chunk_id"],
                f"REFERENCES_UBO_{ubo_ent['entity_value']}_{ubo_ent['chunk_id']}",
            )
            edges.append({
                "edge_id": edge_id,
                "source_node": org_ent["chunk_id"],
                "target_node": ubo_ent["chunk_id"],
                "relationship_type": RelationshipType.REFERENCES_UBO,
                "metadata": {
                    "ubo_value": ubo_ent["entity_value"],
                    "org_value": org_ent["entity_value"],
                },
            })
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Batch embedding + entity extraction + graph construction
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task
def batch_embed_and_store(
    child_chunks: List[Dict[str, Any]],
    customer_id: str,
    doc_id: str,
):
    db = SessionLocal()
    try:
        # ── Dense embeddings (batched, passage-prefixed) ──────────────────────
        embed_texts = [c["embed_text"] for c in child_chunks]
        embeddings = embedding_service.get_passage_embeddings(
            embed_texts, batch_size=settings.EMBEDDING_BATCH_SIZE
        )

        # ── Sparse (SPLADE) embeddings ────────────────────────────────────────
        raw_texts = [c["text"] for c in child_chunks]
        sparse_vectors = embedding_service.get_sparse_embeddings(raw_texts)

        # ── NER with cross-chunk sliding window context ───────────────────────
        chunk_entities_list = extract_entities_windowed(raw_texts, window=2)

        # ── Build insert value lists ──────────────────────────────────────────
        chunk_values = []
        entity_values: List[Dict[str, Any]] = []
        edge_values: List[Dict[str, Any]] = []

        for chunk_data, embedding, sparse_vec, chunk_entities in zip(
            child_chunks, embeddings, sparse_vectors, chunk_entities_list
        ):
            chunk_id = uuid.UUID(chunk_data["chunk_id"])
            segment_id = uuid.UUID(chunk_data["segment_id"])

            chunk_values.append({
                "chunk_id": chunk_id,
                "segment_id": segment_id,
                "text_content": chunk_data["text"],
                "dense_vector": embedding,
                "sparse_vector": sparse_vec,
                "chunk_metadata": chunk_data.get("chunk_metadata", {}),
            })

            # Entity extraction
            for ent in chunk_entities:
                ent_type = _ENTITY_TYPE_MAP.get(ent["type"])
                if ent_type is None:
                    continue
                ent_uuid = uuid.uuid5(chunk_id, f"{ent_type}_{ent['value']}")
                entity_values.append({
                    "entity_id": ent_uuid,
                    "chunk_id": chunk_id,
                    "entity_type": ent_type,
                    "entity_value": ent["value"],
                    "confidence": ent.get("confidence"),
                    "extraction_method": ent.get("method"),
                })

            # CHILD_OF edge (chunk → parent segment)
            edge_uuid = uuid.uuid5(chunk_id, f"CHILD_OF_{segment_id}")
            edge_values.append({
                "edge_id": edge_uuid,
                "source_node": chunk_id,
                "target_node": segment_id,
                "relationship_type": RelationshipType.CHILD_OF,
                "metadata": None,
            })

        # ── Graph edges derived from entity relationships ─────────────────────
        edge_values.extend(_build_same_address_edges(entity_values))
        edge_values.extend(_build_contradicts_edges(entity_values))
        edge_values.extend(_build_ubo_edges(entity_values))

        # ── Bulk inserts ──────────────────────────────────────────────────────
        if chunk_values:
            stmt = insert(SemanticChildChunk).values(chunk_values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["chunk_id"],
                set_={
                    "text_content": stmt.excluded.text_content,
                    "dense_vector": stmt.excluded.dense_vector,
                    "sparse_vector": stmt.excluded.sparse_vector,
                    "chunk_metadata": stmt.excluded.chunk_metadata,
                },
            )
            db.execute(stmt)

        if entity_values:
            stmt_ent = insert(ExtractedEntity).values(entity_values)
            stmt_ent = stmt_ent.on_conflict_do_nothing(index_elements=["entity_id"])
            db.execute(stmt_ent)

        if edge_values:
            stmt_edge = insert(KnowledgeGraphEdge).values(edge_values)
            stmt_edge = stmt_edge.on_conflict_do_nothing(index_elements=["edge_id"])
            db.execute(stmt_edge)

        # ── Build processing_summary for status polling ────────────────────────
        entity_type_counts: Dict[str, int] = {}
        for ev in entity_values:
            key = str(ev["entity_type"].value) if hasattr(ev["entity_type"], "value") else str(ev["entity_type"])
            entity_type_counts[key] = entity_type_counts.get(key, 0) + 1

        processing_summary = {
            "segments": len({c["segment_id"] for c in chunk_values}),
            "chunks": len(chunk_values),
            "edges": len(edge_values),
            "entities": entity_type_counts,
        }

        # ── Mark document as processed ────────────────────────────────────────
        db.query(RawDocument).filter(
            RawDocument.doc_id == uuid.UUID(doc_id)
        ).update({
            "processed_at": datetime.now(timezone.utc),
            "processing_summary": processing_summary,
        })

        db.commit()

        # ── Cache invalidation (P0.5) ─────────────────────────────────────────
        # Publish a Redis message so the retrieve endpoint can flush stale cache
        # entries for this customer.
        try:
            import redis as redis_lib
            r = redis_lib.Redis(
                host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=2
            )
            # Pattern-delete all cached query results for this customer
            pattern = f"query:*_{customer_id}_*"
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=pattern, count=200)
                if keys:
                    r.delete(*keys)
                if cursor == 0:
                    break
            r.publish(f"doc_updated:{customer_id}", doc_id)
        except Exception as exc:
            logger.warning("Cache invalidation failed (non-fatal): %s", exc)

        # ── Entity canonicalization (async, non-blocking) ──────────────────────
        try:
            from app.services.canonicalization_service import canonicalize_entities_task
            canonicalize_entities_task.delay(doc_id)
        except Exception as exc:
            logger.warning("Could not dispatch canonicalization task (non-fatal): %s", exc)

    except Exception as exc:
        db.rollback()
        # Write error summary so status endpoint reports "failed"
        try:
            db.query(RawDocument).filter(
                RawDocument.doc_id == uuid.UUID(doc_id)
            ).update({"processing_summary": {"error": str(exc)}})
            db.commit()
        except Exception:
            pass
        raise exc
    finally:
        db.close()
