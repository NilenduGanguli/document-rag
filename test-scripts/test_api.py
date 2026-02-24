#!/usr/bin/env python3
"""
test_api.py
===========
End-to-end test suite for the document-rag API.

Tests:
  1. Health check (GET /)
  2. Ingest — one file per format type from test-data/ (via manifest.json)
  3. Ingest — error cases (bad extension, empty file, missing field, unsupported MIME)
  4. Status polling — waits for each ingested document to reach completed/failed
  5. Retrieve — queries against ingested documents (entity lookup, semantic, top_k, routing_hints)
  6. Retrieve — edge cases (no results expected, large top_k, repeated query for cache hit)

Usage:
  # Full run (API + worker must be running)
  docker compose up -d
  python test-scripts/test_api.py

  # Custom URL / customer
  python test-scripts/test_api.py --api-url http://localhost:8000 --customer-id smoke-001

  # Skip ingestion (use pre-ingested doc_ids from a previous run)
  python test-scripts/test_api.py --skip-ingest

  # Only ingest specific form types
  python test-scripts/test_api.py --forms "10-K,DOCX,PNG"

  # Ingest k random documents from test-data (different selection each run)
  python test-scripts/test_api.py --sample 3

  # Combine: k random docs filtered to a specific form type
  python test-scripts/test_api.py --sample 2 --forms "10-K,DOCX"

  # Adjust how long to wait for pipeline completion (default: 300 s)
  python test-scripts/test_api.py --timeout 120
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).parent.resolve()
REPO_ROOT     = SCRIPT_DIR.parent
TEST_DATA_ROOT = REPO_ROOT / "test-data"
MANIFEST_PATH  = TEST_DATA_ROOT / "manifest.json"

# ── ANSI colours ───────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _green(t: str)  -> str: return _c("32", t)
def _red(t: str)    -> str: return _c("31", t)
def _yellow(t: str) -> str: return _c("33", t)
def _bold(t: str)   -> str: return _c("1", t)
def _cyan(t: str)   -> str: return _c("36", t)


# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    elapsed_ms: Optional[int] = None

_results: List[TestResult] = []


def _record(name: str, passed: bool, detail: str = "", elapsed_ms: Optional[int] = None) -> bool:
    _results.append(TestResult(name, passed, detail, elapsed_ms))
    status   = _green("PASS") if passed else _red("FAIL")
    elapsed  = f"  {elapsed_ms} ms" if elapsed_ms is not None else ""
    detail_s = f"  — {detail}" if detail else ""
    print(f"  [{status}] {name}{detail_s}{elapsed}")
    return passed


# ── HTTP helpers ───────────────────────────────────────────────────────────────

_session = requests.Session()

def _get(api_url: str, path: str, **kwargs) -> requests.Response:
    return _session.get(f"{api_url.rstrip('/')}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)

def _post(api_url: str, path: str, **kwargs) -> requests.Response:
    return _session.post(f"{api_url.rstrip('/')}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Health check
# ══════════════════════════════════════════════════════════════════════════════

def test_health(api_url: str) -> bool:
    print(_bold("\n── 1. Health check ──────────────────────────────────────────────────────────"))
    t0 = time.perf_counter()
    try:
        resp = _get(api_url, "/")
        elapsed = round((time.perf_counter() - t0) * 1000)
        ok = resp.status_code == 200
        detail = resp.json().get("message", "") if ok else f"HTTP {resp.status_code}"
        return _record("GET /", ok, detail, elapsed)
    except Exception as exc:
        return _record("GET /", False, str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ingest — one file per format type
# ══════════════════════════════════════════════════════════════════════════════

# Mapping: file extension → representative test for the suite
# We pick one file per extension type to keep things quick; all types covered.
_EXT_PRIORITY = [".pdf", ".xlsx", ".docx", ".png", ".tiff"]

def _pick_one_per_ext(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """From manifest, return one entry per file extension (first occurrence wins)."""
    seen_exts: set = set()
    picked: List[Dict[str, Any]] = []
    for entry in manifest:
        ext = Path(entry["file_path"]).suffix.lower()
        if ext not in seen_exts:
            seen_exts.add(ext)
            picked.append(entry)
    # Ensure ordering by priority list
    ordered = [e for ex in _EXT_PRIORITY for e in picked if Path(e["file_path"]).suffix.lower() == ex]
    # Append anything not in priority list
    ordered += [e for e in picked if Path(e["file_path"]).suffix.lower() not in _EXT_PRIORITY]
    return ordered


def _pick_random(manifest: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Return k documents chosen at random from manifest (no fixed seed — differs each run)."""
    k = min(k, len(manifest))
    return random.sample(manifest, k)


def _ingest_file(
    api_url: str,
    customer_id: str,
    file_path: Path,
    form_type: str,
    processing_directives: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], int]:
    """
    POST /api/v1/ingest.
    Returns (doc_id, http_status_code).  doc_id is None on failure.
    """
    t0 = time.perf_counter()
    with open(file_path, "rb") as fh:
        data: Dict[str, Any] = {"customer_id": customer_id}
        if processing_directives:
            data["processing_directives_raw"] = json.dumps(processing_directives)
        resp = _post(
            api_url, "/api/v1/ingest",
            files={"file": (file_path.name, fh)},
            data=data,
            timeout=60,
        )
    elapsed = round((time.perf_counter() - t0) * 1000)
    if resp.status_code in (200, 202):
        doc_id = resp.json().get("doc_id")
        return doc_id, elapsed
    return None, elapsed


def _poll_status(api_url: str, doc_id: str, deadline: float) -> Dict[str, Any]:
    """Poll GET /api/v1/ingest/{doc_id}/status until terminal state or deadline."""
    while time.time() < deadline:
        time.sleep(3)
        try:
            resp = _get(api_url, f"/api/v1/ingest/{doc_id}/status")
            if resp.status_code == 200:
                data = resp.json()
                st = data.get("status", "queued")
                if st in ("completed", "failed"):
                    return data
        except Exception:
            pass
    return {"status": "timeout", "error": "Pipeline did not finish within timeout."}


def test_ingest(
    api_url: str,
    customer_id: str,
    manifest: List[Dict[str, Any]],
    form_filter: set,
    pipeline_timeout: int,
    picks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    """
    Ingest documents.  If picks is provided (e.g. from --sample), those are used
    directly; otherwise falls back to one file per format type.
    Returns {form_type: doc_id} for use in retrieval tests.
    """
    if picks is None:
        picks = _pick_one_per_ext(manifest)
        if form_filter:
            picks = [p for p in picks if p["form_type"] in form_filter]
        header = "one file per format type"
    else:
        header = f"{len(picks)} randomly sampled file(s)"

    print(_bold(f"\n── 2. Ingest ({header}) {'─' * max(0, 57 - len(header))}"))

    ingested: Dict[str, Optional[str]] = {}

    for entry in picks:
        fp = REPO_ROOT / entry["file_path"]
        ft = entry["form_type"]
        label = f"ingest {Path(entry['file_path']).name} ({ft})"

        if not fp.exists():
            _record(label, False, "file not found — run prepare_test_documents.py first")
            continue

        doc_id, ingest_ms = _ingest_file(api_url, customer_id, fp, ft)
        if not doc_id:
            _record(label, False, "ingest request failed", ingest_ms)
            continue

        # Poll for completion
        deadline = time.time() + pipeline_timeout
        result = _poll_status(api_url, doc_id, deadline)
        total_ms = ingest_ms  # (poll time not tracked separately for brevity)

        status = result.get("status")
        chunks = result.get("semantic_chunks_created")
        segments = result.get("layout_segments_created")
        error = result.get("error")

        if status == "completed":
            detail = f"doc_id={doc_id[:8]}…  segments={segments}  chunks={chunks}"
            _record(label, True, detail)
            ingested[ft] = doc_id
        elif status == "failed":
            _record(label, False, f"pipeline failed: {error}")
            ingested[ft] = doc_id  # still useful for status endpoint tests
        else:
            _record(label, False, f"status={status}  error={error}")

    return ingested


# ══════════════════════════════════════════════════════════════════════════════
# 3. Ingest — error cases
# ══════════════════════════════════════════════════════════════════════════════

def test_ingest_errors(api_url: str, customer_id: str) -> None:
    print(_bold("\n── 3. Ingest error cases ─────────────────────────────────────────────────────"))

    # 3a. Unsupported file extension
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        tf.write(b"col1,col2\n1,2\n")
        tmp_csv = tf.name
    try:
        t0 = time.perf_counter()
        resp = _post(
            api_url, "/api/v1/ingest",
            files={"file": ("data.csv", open(tmp_csv, "rb"))},
            data={"customer_id": customer_id},
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        _record(
            "ingest rejects unsupported extension (.csv)",
            resp.status_code == 400,
            f"HTTP {resp.status_code}",
            elapsed,
        )
    finally:
        os.unlink(tmp_csv)

    # 3b. Empty file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tmp_empty = tf.name  # zero bytes
    try:
        t0 = time.perf_counter()
        resp = _post(
            api_url, "/api/v1/ingest",
            files={"file": ("empty.pdf", open(tmp_empty, "rb"))},
            data={"customer_id": customer_id},
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        _record(
            "ingest rejects empty file",
            resp.status_code == 400,
            f"HTTP {resp.status_code}",
            elapsed,
        )
    finally:
        os.unlink(tmp_empty)

    # 3c. Missing customer_id
    sample_pdf = next(
        (REPO_ROOT / e["file_path"]
         for e in _load_manifest()
         if e["format"] == "pdf" and (REPO_ROOT / e["file_path"]).exists()),
        None,
    )
    if sample_pdf:
        t0 = time.perf_counter()
        resp = _post(
            api_url, "/api/v1/ingest",
            files={"file": (sample_pdf.name, open(sample_pdf, "rb"))},
            data={},  # no customer_id
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        _record(
            "ingest rejects missing customer_id",
            resp.status_code in (400, 422),
            f"HTTP {resp.status_code}",
            elapsed,
        )

    # 3d. MIME mismatch — .pdf extension but PNG content
    sample_png = TEST_DATA_ROOT / "images" / "passport.png"
    if sample_png.exists():
        t0 = time.perf_counter()
        resp = _post(
            api_url, "/api/v1/ingest",
            files={"file": ("fake.pdf", open(sample_png, "rb"), "application/pdf")},
            data={"customer_id": customer_id},
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        _record(
            "ingest rejects MIME mismatch (.pdf ext / PNG bytes)",
            resp.status_code == 400,
            f"HTTP {resp.status_code}",
            elapsed,
        )

    # 3e. Status for nonexistent doc_id
    fake_id = "00000000-0000-0000-0000-000000000000"
    t0 = time.perf_counter()
    resp = _get(api_url, f"/api/v1/ingest/{fake_id}/status")
    elapsed = round((time.perf_counter() - t0) * 1000)
    _record(
        "status 404 for unknown doc_id",
        resp.status_code == 404,
        f"HTTP {resp.status_code}",
        elapsed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Status endpoint — field validation on a completed document
# ══════════════════════════════════════════════════════════════════════════════

def test_status_fields(api_url: str, ingested: Dict[str, Optional[str]]) -> None:
    print(_bold("\n── 4. Status response field validation ──────────────────────────────────────"))

    completed_doc_id = next(
        (doc_id for doc_id in ingested.values() if doc_id), None
    )
    if not completed_doc_id:
        _record("status field validation", False, "no ingested doc_id available")
        return

    resp = _get(api_url, f"/api/v1/ingest/{completed_doc_id}/status")
    if resp.status_code != 200:
        _record("status field validation", False, f"HTTP {resp.status_code}")
        return

    data = resp.json()
    required_fields = {"doc_id", "status"}
    missing = required_fields - set(data.keys())
    _record(
        "status response contains required fields",
        len(missing) == 0,
        f"missing: {missing}" if missing else f"status={data.get('status')}",
    )

    if data.get("status") == "completed":
        _record(
            "completed doc has non-null semantic_chunks_created",
            data.get("semantic_chunks_created") is not None
            and data["semantic_chunks_created"] > 0,
            f"chunks={data.get('semantic_chunks_created')}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Retrieve
# ══════════════════════════════════════════════════════════════════════════════

def _retrieve(
    api_url: str,
    query: str,
    customer_id: Optional[str] = None,
    top_k: int = 5,
    routing_hints: Optional[List[str]] = None,
) -> Tuple[requests.Response, int]:
    payload: Dict[str, Any] = {"query": query, "top_k": top_k}
    if customer_id:
        payload["customer_id"] = customer_id
    if routing_hints:
        payload["routing_hints"] = routing_hints
    t0 = time.perf_counter()
    resp = _post(api_url, "/api/v1/retrieve", json=payload)
    elapsed = round((time.perf_counter() - t0) * 1000)
    return resp, elapsed


def test_retrieve(api_url: str, customer_id: str, ingested: Dict[str, Optional[str]]) -> None:
    print(_bold("\n── 5. Retrieve ───────────────────────────────────────────────────────────────"))

    has_ingested = any(v for v in ingested.values())

    # ── 5a. Basic semantic query against Apple 10-K ──────────────────────────
    resp, ms = _retrieve(api_url, "What is Apple's annual revenue?", customer_id)
    ok = resp.status_code == 200
    chunks = len(resp.json().get("retrieved_chunks", [])) if ok else 0
    _record(
        "retrieve: Apple annual revenue query",
        ok and (chunks > 0 or not has_ingested),
        f"HTTP {resp.status_code}  chunks={chunks}" if ok else f"HTTP {resp.status_code}",
        ms,
    )

    # ── 5b. Query with custom top_k ──────────────────────────────────────────
    resp, ms = _retrieve(api_url, "risk factors", customer_id, top_k=3)
    ok = resp.status_code == 200
    if ok:
        chunks = resp.json().get("retrieved_chunks", [])
        _record(
            "retrieve: top_k=3 respected",
            len(chunks) <= 3,
            f"returned {len(chunks)} chunks",
            ms,
        )
    else:
        _record("retrieve: top_k=3 respected", False, f"HTTP {resp.status_code}", ms)

    # ── 5c. Entity-oriented query (KYC document) ─────────────────────────────
    resp, ms = _retrieve(
        api_url,
        "What is the IBAN and passport number for Jane Doe?",
        customer_id,
    )
    ok = resp.status_code == 200
    chunks = len(resp.json().get("retrieved_chunks", [])) if ok else 0
    _record(
        "retrieve: KYC entity query (IBAN / passport)",
        ok and (chunks > 0 or not has_ingested),
        f"HTTP {resp.status_code}  chunks={chunks}" if ok else f"HTTP {resp.status_code}",
        ms,
    )

    # ── 5d. Routing hint — IBAN bypass ───────────────────────────────────────
    resp, ms = _retrieve(
        api_url,
        "DE89 3704 0044 0532 0130 00",
        customer_id,
        routing_hints=["IBAN"],
    )
    ok = resp.status_code == 200
    if ok:
        data = resp.json()
        decision = data.get("router_decision", "")
        chunks = len(data.get("retrieved_chunks", []))
        _record(
            "retrieve: routing_hints=[IBAN] accepted",
            ok,
            f"decision={decision}  chunks={chunks}",
            ms,
        )
    else:
        _record("retrieve: routing_hints=[IBAN] accepted", False, f"HTTP {resp.status_code}", ms)

    # ── 5e. SEC filing query — Microsoft 10-Q ────────────────────────────────
    resp, ms = _retrieve(api_url, "Microsoft quarterly earnings cloud revenue", customer_id)
    ok = resp.status_code == 200
    chunks = len(resp.json().get("retrieved_chunks", [])) if ok else 0
    _record(
        "retrieve: Microsoft 10-Q query",
        ok and (chunks > 0 or not has_ingested),
        f"HTTP {resp.status_code}  chunks={chunks}" if ok else f"HTTP {resp.status_code}",
        ms,
    )

    # ── 5f. FDIC / banking data query ────────────────────────────────────────
    resp, ms = _retrieve(api_url, "bank assets deposits net income", customer_id)
    ok = resp.status_code == 200
    _record(
        "retrieve: FDIC banking data query",
        ok,
        f"HTTP {resp.status_code}  chunks={len(resp.json().get('retrieved_chunks', [])) if ok else '?'}",
        ms,
    )

    # ── 5g. Passport / image OCR query ───────────────────────────────────────
    resp, ms = _retrieve(api_url, "passport expiry nationality", customer_id)
    ok = resp.status_code == 200
    _record(
        "retrieve: passport / image OCR query",
        ok,
        f"HTTP {resp.status_code}  chunks={len(resp.json().get('retrieved_chunks', [])) if ok else '?'}",
        ms,
    )

    # ── 5h. Repeated query (should serve from cache on second hit) ───────────
    q = "beneficial ownership UBO Delaware LLC"
    _, ms1 = _retrieve(api_url, q, customer_id)
    resp2, ms2 = _retrieve(api_url, q, customer_id)
    ok = resp2.status_code == 200
    _record(
        "retrieve: repeated query returns 200 (cache path)",
        ok,
        f"1st={ms1} ms  2nd={ms2} ms",
        ms2,
    )

    # ── 5i. Stage latencies present in response ───────────────────────────────
    resp, ms = _retrieve(api_url, "annual report executive compensation", customer_id)
    if resp.status_code == 200:
        latencies = resp.json().get("stage_latencies") or {}
        has_total = "total_ms" in latencies
        _record(
            "retrieve: response includes stage_latencies.total_ms",
            has_total,
            f"keys={list(latencies.keys())}",
        )
    else:
        _record("retrieve: response includes stage_latencies.total_ms", False, f"HTTP {resp.status_code}")

    # ── 5j. audit_metadata present ───────────────────────────────────────────
    resp, ms = _retrieve(api_url, "company registration tax ID", customer_id)
    if resp.status_code == 200:
        meta = resp.json().get("audit_metadata") or {}
        has_fields = "reranker_model" in meta and "hybrid_search_triggered" in meta
        _record(
            "retrieve: response includes audit_metadata",
            has_fields,
            f"keys={list(meta.keys())}",
        )
    else:
        _record("retrieve: response includes audit_metadata", False, f"HTTP {resp.status_code}")

    # ── 5k. No customer_id (global search) ───────────────────────────────────
    resp, ms = _retrieve(api_url, "securities filing proxy", customer_id=None)
    _record(
        "retrieve: query without customer_id returns 200",
        resp.status_code == 200,
        f"HTTP {resp.status_code}",
        ms,
    )

    # ── 5l. Large top_k ───────────────────────────────────────────────────────
    resp, ms = _retrieve(api_url, "financial statements", customer_id, top_k=50)
    ok = resp.status_code == 200
    _record(
        "retrieve: large top_k=50 accepted",
        ok,
        f"HTTP {resp.status_code}  chunks={len(resp.json().get('retrieved_chunks', [])) if ok else '?'}",
        ms,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Retrieve — error / edge cases
# ══════════════════════════════════════════════════════════════════════════════

def test_retrieve_errors(api_url: str, customer_id: str) -> None:
    print(_bold("\n── 6. Retrieve error / edge cases ────────────────────────────────────────────"))

    # 6a. Empty query string
    resp, ms = _retrieve(api_url, "", customer_id)
    # API may return 200 with no results OR 422 — both are acceptable, just not 500
    _record(
        "retrieve: empty query does not 500",
        resp.status_code != 500,
        f"HTTP {resp.status_code}",
        ms,
    )

    # 6b. Very long query (stress test)
    long_q = ("financial risk assessment " * 40).strip()
    resp, ms = _retrieve(api_url, long_q, customer_id)
    _record(
        "retrieve: very long query (960 chars) returns 200",
        resp.status_code == 200,
        f"HTTP {resp.status_code}",
        ms,
    )

    # 6c. Invalid JSON body (raw POST)
    t0 = time.perf_counter()
    raw_resp = _session.post(
        f"{api_url.rstrip('/')}/api/v1/retrieve",
        data="not json",
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    elapsed = round((time.perf_counter() - t0) * 1000)
    _record(
        "retrieve: invalid JSON body returns 4xx",
        raw_resp.status_code in (400, 422),
        f"HTTP {raw_resp.status_code}",
        elapsed,
    )

    # 6d. top_k=0
    resp, ms = _retrieve(api_url, "test query", customer_id, top_k=0)
    _record(
        "retrieve: top_k=0 does not 500",
        resp.status_code != 500,
        f"HTTP {resp.status_code}",
        ms,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Per-format ingest coverage (all files in manifest)
# ══════════════════════════════════════════════════════════════════════════════

def test_ingest_all_formats(
    api_url: str,
    customer_id: str,
    manifest: List[Dict[str, Any]],
    form_filter: set,
    pipeline_timeout: int,
    already_ingested: Dict[str, Optional[str]],
) -> None:
    """Ingest every remaining file in the manifest (skipping those already tested)."""
    print(_bold("\n── 7. Ingest remaining files from manifest ───────────────────────────────────"))

    ingested_fts = set(already_ingested.keys())
    remaining = [
        e for e in manifest
        if e["form_type"] not in ingested_fts
        and (not form_filter or e["form_type"] in form_filter)
    ]

    if not remaining:
        print(_yellow("  (all manifest form types already covered above)"))
        return

    for entry in remaining:
        fp = REPO_ROOT / entry["file_path"]
        ft = entry["form_type"]
        label = f"ingest {Path(entry['file_path']).name} ({ft})"

        if not fp.exists():
            _record(label, False, "file not found")
            continue

        doc_id, _ = _ingest_file(api_url, customer_id, fp, ft)
        if not doc_id:
            _record(label, False, "ingest request failed")
            continue

        deadline = time.time() + pipeline_timeout
        result = _poll_status(api_url, doc_id, deadline)
        status = result.get("status")
        chunks = result.get("semantic_chunks_created")
        error  = result.get("error")

        if status == "completed":
            _record(label, True, f"doc_id={doc_id[:8]}…  chunks={chunks}")
        elif status == "failed":
            _record(label, False, f"pipeline failed: {error}")
        else:
            _record(label, False, f"status={status}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Processing directives
# ══════════════════════════════════════════════════════════════════════════════

def test_processing_directives(api_url: str, customer_id: str) -> None:
    print(_bold("\n── 8. Processing directives ─────────────────────────────────────────────────"))

    sample_pdf = next(
        (REPO_ROOT / e["file_path"]
         for e in _load_manifest()
         if e["format"] == "pdf" and (REPO_ROOT / e["file_path"]).exists()),
        None,
    )
    if not sample_pdf:
        _record("processing_directives: force_ocr=false", False, "no PDF available")
        return

    # 8a. force_ocr=false (default) — should accept without error
    t0 = time.perf_counter()
    doc_id, ms = _ingest_file(
        api_url, customer_id, sample_pdf, "test-directives",
        processing_directives={"force_ocr": False, "ocr_provider": "gpt4v"},
    )
    _record(
        "processing_directives: force_ocr=false accepted",
        doc_id is not None,
        f"doc_id={doc_id[:8]}…" if doc_id else "no doc_id",
        ms,
    )

    # 8b. Invalid directives JSON string — API should fall back to defaults, not crash
    with open(sample_pdf, "rb") as fh:
        t0 = time.perf_counter()
        resp = _post(
            api_url, "/api/v1/ingest",
            files={"file": (sample_pdf.name, fh)},
            data={
                "customer_id": customer_id,
                "processing_directives_raw": "not-valid-json",
            },
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
    _record(
        "processing_directives: bad JSON falls back gracefully (202 accepted)",
        resp.status_code in (200, 202),
        f"HTTP {resp.status_code}",
        elapsed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_manifest() -> List[Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text())


def _print_summary() -> int:
    """Print final summary table. Returns number of failures."""
    total   = len(_results)
    passed  = sum(1 for r in _results if r.passed)
    failed  = total - passed

    print(_bold("\n" + "═" * 80))
    print(_bold("TEST SUMMARY"))
    print("═" * 80)
    print(f"  Total : {total}")
    print(f"  {_green('Passed')}: {passed}")
    if failed:
        print(f"  {_red('Failed')}: {failed}")
        print()
        print(_red("Failed tests:"))
        for r in _results:
            if not r.passed:
                print(f"    • {r.name}")
                if r.detail:
                    print(f"      {r.detail}")
    print("═" * 80)
    return failed


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end API test suite for document-rag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-url",     default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--customer-id", default="test-suite-001",        help="Customer ID for test documents")
    parser.add_argument("--timeout",     type=int, default=300,            help="Pipeline completion timeout in seconds")
    parser.add_argument("--forms",       default="",                       help="Comma-separated form types to restrict ingest tests")
    parser.add_argument("--skip-ingest", action="store_true",              help="Skip ingestion tests (use for retrieve-only runs)")
    parser.add_argument("--skip-retrieve", action="store_true",            help="Skip retrieval tests")
    parser.add_argument("--sample",      type=int, default=None,           help="Ingest k randomly chosen documents instead of one per format type")
    args = parser.parse_args()

    form_filter = {f.strip() for f in args.forms.split(",") if f.strip()}
    api_url     = args.api_url.rstrip("/")
    customer_id = args.customer_id

    print(_bold(_cyan(f"\n{'═' * 80}")))
    print(_bold(_cyan(f"  document-rag API Test Suite")))
    print(_bold(_cyan(f"  API: {api_url}   customer: {customer_id}")))
    print(_bold(_cyan(f"{'═' * 80}")))

    manifest = _load_manifest()
    if not manifest:
        print(_yellow(
            "\n[warn] test-data/manifest.json not found or empty.\n"
            "       Run: python test-scripts/prepare_test_documents.py --skip-ingest\n"
            "       to download test data before running this suite.\n"
        ))

    # 1. Health
    api_reachable = test_health(api_url)
    if not api_reachable:
        print(_red("\n[error] API not reachable — aborting remaining tests."))
        sys.exit(_print_summary() > 0)

    ingested: Dict[str, Optional[str]] = {}

    if not args.skip_ingest and manifest:
        # Build the pick list for ingest step 2
        if args.sample is not None:
            pool = manifest
            if form_filter:
                pool = [e for e in manifest if e["form_type"] in form_filter]
            if not pool:
                print(_yellow("\n[warn] --sample: no manifest entries match the given --forms filter."))
                sample_picks: Optional[List[Dict[str, Any]]] = []
            else:
                k = min(args.sample, len(pool))
                sample_picks = random.sample(pool, k)
                names = ", ".join(Path(e["file_path"]).name for e in sample_picks)
                print(_yellow(f"\n[info] --sample {args.sample}: selected {k} document(s) at random: {names}"))
        else:
            sample_picks = None  # test_ingest will use _pick_one_per_ext

        # 2. Ingest
        ingested = test_ingest(api_url, customer_id, manifest, form_filter, args.timeout, sample_picks)

        # 3. Ingest error cases
        test_ingest_errors(api_url, customer_id)

        # 4. Status field validation
        test_status_fields(api_url, ingested)

        # 7. Ingest remaining manifest entries (skipped when --sample is active)
        if args.sample is None:
            test_ingest_all_formats(api_url, customer_id, manifest, form_filter, args.timeout, ingested)

        # 8. Processing directives
        test_processing_directives(api_url, customer_id)

    elif args.skip_ingest:
        print(_yellow("\n[info] --skip-ingest: ingestion tests skipped."))
        # Still run error cases that don't depend on successful ingest
        test_ingest_errors(api_url, customer_id)

    if not args.skip_retrieve:
        # 5. Retrieve
        test_retrieve(api_url, customer_id, ingested)

        # 6. Retrieve error / edge cases
        test_retrieve_errors(api_url, customer_id)

    failures = _print_summary()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
