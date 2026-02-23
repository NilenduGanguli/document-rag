"""
KYC Document RAG — Streamlit Frontend
End-to-end pipeline: Upload → S3 → OCR → Chunk → Embed → Retrieve
"""
import time
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import os

# ── Page setup (must be the very first st.* call) ─────────────────────────────
st.set_page_config(
    page_title="KYC RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://api:8000/api/v1")
DB_URI = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'postgres')}"
    f"@{os.getenv('POSTGRES_SERVER', 'db')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'kyc_rag')}"
)
POLL_INTERVAL = 2   # seconds between DB polls while a job is running

@st.cache_resource
def get_engine():
    return create_engine(DB_URI, pool_pre_ping=True)

engine = get_engine()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 KYC RAG")
    st.caption("End-to-end document intelligence pipeline")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📤 Ingest & Monitor", "🗄️ Pipeline Explorer", "🔎 Retrieval"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Services")
    try:
        r = requests.get(f"{API_URL.rsplit('/api', 1)[0]}/", timeout=2)
        st.success("API  ✅ online")
    except Exception:
        st.error("API  ❌ offline")


# ── Helpers ───────────────────────────────────────────────────────────────────
def query_db(sql: str, params: dict | None = None):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def scalar_db(sql: str, params: dict | None = None):
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar() or 0


def pipeline_status(doc_id: str) -> dict:
    """Return counts for each pipeline stage for a given doc_id."""
    base = {"doc_id": doc_id}
    try:
        base["segments"] = scalar_db(
            "SELECT COUNT(*) FROM parsed_layout_segment WHERE doc_id = :d",
            {"d": doc_id},
        )
        base["chunks"] = scalar_db(
            """SELECT COUNT(*) FROM semantic_child_chunk scc
               JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
               WHERE pls.doc_id = :d""",
            {"d": doc_id},
        )
        base["entities"] = scalar_db(
            """SELECT COUNT(*) FROM extracted_entity ee
               JOIN semantic_child_chunk scc ON ee.chunk_id = scc.chunk_id
               JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
               WHERE pls.doc_id = :d""",
            {"d": doc_id},
        )
        base["embedded"] = scalar_db(
            """SELECT COUNT(*) FROM semantic_child_chunk scc
               JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
               WHERE pls.doc_id = :d AND scc.dense_vector IS NOT NULL""",
            {"d": doc_id},
        )
    except Exception as e:
        base["error"] = str(e)
    return base


def stage_label(status: dict) -> str:
    if status.get("embedded", 0) > 0:
        return "done"
    if status.get("entities", 0) > 0:
        return "embedding"
    if status.get("chunks", 0) > 0:
        return "nlp"
    if status.get("segments", 0) > 0:
        return "chunking"
    return "ocr"


def render_pipeline_progress(status: dict, storage_uri: str):
    stage = stage_label(status)

    stages = [
        ("upload",    "☁️",  "Upload to S3",       True,                              storage_uri or ""),
        ("ocr",       "🔬",  "OCR & Layout Parse", status.get("segments", 0) > 0,    f"{status.get('segments', 0)} segment(s)"),
        ("chunking",  "✂️",  "Chunking",           status.get("chunks", 0) > 0,      f"{status.get('chunks', 0)} chunk(s)"),
        ("nlp",       "🏷️",  "Entity Extraction",  status.get("entities", 0) > 0,    f"{status.get('entities', 0)} entit(ies)"),
        ("embedding", "🧮",  "Embedding & Index",  status.get("embedded", 0) > 0,    f"{status.get('embedded', 0)} vector(s)"),
    ]

    cols = st.columns(len(stages))
    for col, (key, icon, label, done, detail) in zip(cols, stages):
        with col:
            if done:
                st.markdown(
                    f"""<div style='text-align:center;padding:12px;border-radius:10px;
                    background:#1a3a1a;border:1px solid #2d6a2d'>
                    <div style='font-size:28px'>{icon}</div>
                    <div style='color:#4CAF50;font-weight:600;margin:4px 0'>{label}</div>
                    <div style='color:#aaa;font-size:12px'>{detail}</div></div>""",
                    unsafe_allow_html=True,
                )
            elif key == stage:
                st.markdown(
                    f"""<div style='text-align:center;padding:12px;border-radius:10px;
                    background:#1a1a3a;border:1px solid #4444cc'>
                    <div style='font-size:28px'>{icon}</div>
                    <div style='color:#5588ff;font-weight:600;margin:4px 0'>{label}</div>
                    <div style='color:#aaa;font-size:12px'>Processing…</div></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div style='text-align:center;padding:12px;border-radius:10px;
                    background:#1a1a1a;border:1px solid #333'>
                    <div style='font-size:28px;opacity:.4'>{icon}</div>
                    <div style='color:#555;font-weight:600;margin:4px 0'>{label}</div>
                    <div style='color:#444;font-size:12px'>Waiting</div></div>""",
                    unsafe_allow_html=True,
                )


def render_entities(doc_id: str):
    try:
        df = query_db(
            """SELECT ee.entity_type, ee.entity_value
               FROM extracted_entity ee
               JOIN semantic_child_chunk scc ON ee.chunk_id = scc.chunk_id
               JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
               WHERE pls.doc_id = :d
               ORDER BY ee.entity_type""",
            {"d": doc_id},
        )
        if not df.empty:
            st.markdown("#### 🏷️ Extracted Entities")
            for etype, group in df.groupby("entity_type"):
                values = ", ".join(f"`{v}`" for v in group["entity_value"].tolist())
                st.markdown(f"**{etype}** — {values}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INGEST & MONITOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "📤 Ingest & Monitor":
    st.title("📤 Ingest & Monitor")
    st.caption("Upload a KYC document and watch it move through every pipeline stage in real time.")

    with st.form("ingest_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Document (PDF, PNG, JPEG, TIFF)",
                type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
            )
        with col2:
            customer_id = st.text_input("Customer ID", value="cust-001")
        submitted = st.form_submit_button("🚀 Upload & Ingest", use_container_width=True)

    if submitted:
        if not uploaded_file:
            st.warning("Please select a file.")
        else:
            with st.spinner("Uploading to S3…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest",
                        data={"customer_id": customer_id},
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=30,
                    )
                    if resp.status_code == 202:
                        data = resp.json()
                        st.session_state["active_doc_id"]       = data["doc_id"]
                        st.session_state["active_task_id"]      = data["task_id"]
                        st.session_state["active_storage_uri"]  = data["storage_uri"]
                        st.session_state["active_customer_id"]  = customer_id
                        st.session_state["active_filename"]     = uploaded_file.name
                        st.session_state["poll_start"]          = time.time()
                    else:
                        st.error(f"Ingestion failed ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")

    # ── Live progress monitor ─────────────────────────────────────────────────
    if "active_doc_id" in st.session_state:
        doc_id      = st.session_state["active_doc_id"]
        task_id     = st.session_state["active_task_id"]
        storage_uri = st.session_state["active_storage_uri"]
        filename    = st.session_state.get("active_filename", "document")
        cust        = st.session_state.get("active_customer_id", "")

        st.divider()
        st.markdown(f"### 📄 `{filename}` — customer `{cust}`")
        c1, c2 = st.columns(2)
        c1.caption(f"**doc_id** `{doc_id}`")
        c2.caption(f"**task_id** `{task_id}`")

        status  = pipeline_status(doc_id)
        is_done = status.get("embedded", 0) > 0

        render_pipeline_progress(status, storage_uri)

        if not is_done:
            elapsed = int(time.time() - st.session_state.get("poll_start", time.time()))
            st.info(f"⏳ Processing… (elapsed: {elapsed}s) — auto-refreshing every {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            st.rerun()
        else:
            st.success("✅ Pipeline complete — document is ready for retrieval!")
            render_entities(doc_id)
            if st.button("🔎 Query this document"):
                st.session_state["prefill_customer_id"] = cust
                st.rerun()

    # ── Recent jobs table ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🗒️ Recent Ingestions")
    try:
        recent = query_db(
            """SELECT rd.doc_id, rd.customer_id, rd.file_type,
                      rd.storage_uri,
                      (SELECT COUNT(*) FROM parsed_layout_segment pls
                       WHERE pls.doc_id = rd.doc_id) AS segments,
                      (SELECT COUNT(*) FROM semantic_child_chunk scc
                       JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                       WHERE pls.doc_id = rd.doc_id) AS chunks,
                      (SELECT COUNT(*) FROM extracted_entity ee
                       JOIN semantic_child_chunk scc ON ee.chunk_id = scc.chunk_id
                       JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                       WHERE pls.doc_id = rd.doc_id) AS entities,
                      (SELECT COUNT(*) FROM semantic_child_chunk scc
                       JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                       WHERE pls.doc_id = rd.doc_id AND scc.dense_vector IS NOT NULL) AS embedded
               FROM raw_document rd
               ORDER BY rd.doc_id DESC LIMIT 15"""
        )

        def status_badge(row):
            if row["embedded"] > 0:   return "✅ Done"
            if row["entities"] > 0:   return "🧮 Embedding"
            if row["chunks"] > 0:     return "🏷️ NLP"
            if row["segments"] > 0:   return "✂️ Chunking"
            return "🔬 OCR"

        recent["status"] = recent.apply(status_badge, axis=1)
        recent["storage_uri"] = recent["storage_uri"].str.replace(
            "s3://kyc-documents/", "s3://…/", regex=False
        )
        st.dataframe(
            recent[["doc_id", "customer_id", "file_type", "status",
                    "segments", "chunks", "entities", "embedded", "storage_uri"]],
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        st.warning(f"Could not load recent jobs: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PIPELINE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ Pipeline Explorer":
    st.title("🗄️ Pipeline Explorer")
    st.caption("Browse every layer of the pipeline database.")

    TABLE_META = {
        "raw_document":          {"icon": "📁", "desc": "One row per ingested document"},
        "parsed_layout_segment": {"icon": "📄", "desc": "OCR output — layout blocks per document"},
        "semantic_child_chunk":  {"icon": "✂️",  "desc": "Chunks with dense + sparse vectors"},
        "extracted_entity":      {"icon": "🏷️", "desc": "NER entities linked to chunks"},
        "knowledge_graph_edge":  {"icon": "🕸️", "desc": "Relationships between entity nodes"},
        "retrieval_audit_log":   {"icon": "📋", "desc": "Every retrieval request logged"},
    }

    st.markdown("#### Table Overview")
    cnt_cols = st.columns(len(TABLE_META))
    for col, (tbl, meta) in zip(cnt_cols, TABLE_META.items()):
        try:
            cnt = scalar_db(f"SELECT COUNT(*) FROM {tbl}")
        except Exception:
            cnt = "?"
        col.metric(f"{meta['icon']} {tbl.replace('_', ' ').title()}", cnt)

    st.divider()

    col_left, col_right = st.columns([1, 3])
    with col_left:
        selected_table = st.radio(
            "Select table",
            list(TABLE_META.keys()),
            format_func=lambda t: f"{TABLE_META[t]['icon']} {t}",
        )
        st.caption(TABLE_META[selected_table]["desc"])
        filter_doc = st.text_input("Filter by doc_id (optional)")
        limit = st.slider("Row limit", 10, 500, 50)

    with col_right:
        try:
            if selected_table == "semantic_child_chunk":
                if filter_doc:
                    sql = """SELECT scc.chunk_id, scc.segment_id, scc.text_content
                             FROM semantic_child_chunk scc
                             JOIN parsed_layout_segment pls ON scc.segment_id = pls.segment_id
                             WHERE pls.doc_id = :doc LIMIT :lim"""
                    df = query_db(sql, {"doc": filter_doc, "lim": limit})
                else:
                    df = query_db(f"SELECT chunk_id, segment_id, text_content FROM semantic_child_chunk LIMIT :lim", {"lim": limit})
            elif selected_table in ("raw_document", "parsed_layout_segment") and filter_doc:
                df = query_db(f"SELECT * FROM {selected_table} WHERE doc_id = :doc LIMIT :lim", {"doc": filter_doc, "lim": limit})
            else:
                df = query_db(f"SELECT * FROM {selected_table} LIMIT :lim", {"lim": limit})
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Retrieval":
    st.title("🔎 Retrieval")
    st.caption("Multi-layer gateway: Deterministic Bypass → Hybrid RRF Search → Cross-Encoder Reranking → Graph Traversal")

    default_cust = st.session_state.pop("prefill_customer_id", "")

    with st.form("retrieve_form"):
        query = st.text_input("Query", placeholder="Who is the passport holder and what is their nationality?")
        col1, col2 = st.columns([2, 1])
        with col1:
            customer_id = st.text_input("Customer ID (leave blank for all)", value=default_cust)
        with col2:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("🔍 Retrieve", use_container_width=True)

    if submitted and query:
        payload = {"query": query, "top_k": int(top_k)}
        if customer_id.strip():
            payload["customer_id"] = customer_id.strip()

        with st.spinner("Running retrieval pipeline…"):
            try:
                resp = requests.post(f"{API_URL}/retrieve", json=payload, timeout=60)
            except requests.exceptions.Timeout:
                st.error("Request timed out. The reranker may still be loading — try again in a moment.")
                st.stop()
            except Exception as e:
                st.error(f"Could not reach API: {e}")
                st.stop()

        if resp.status_code != 200:
            st.error(f"Retrieval failed ({resp.status_code}): {resp.text}")
            st.stop()

        # Persist results so checkbox toggles (which cause reruns) don't wipe them
        st.session_state["retrieve_data"] = resp.json()

    # ── Render results (from session state so reruns preserve them) ───────────
    data = st.session_state.get("retrieve_data")
    if data:
        chunks = data.get("retrieved_chunks", [])

        st.divider()
        m1, m2, m3 = st.columns(3)
        router         = data.get("router_decision", "—")
        reranker_score = data.get("confidence_scores", {}).get("reranker_avg", 0)
        m1.metric("Router Decision",    router.replace("_", " ").title())
        m2.metric("Chunks Retrieved",   len(chunks))
        m3.metric("Reranker Avg Score", f"{reranker_score:.4f}")

        if not chunks:
            st.info("No results found. Try a different query or ingest more documents.")
        else:
            st.markdown("#### Retrieved Chunks")
            for i, chunk in enumerate(chunks):
                score  = chunk.get("score", 0)
                text   = chunk.get("text_content", "")
                parent = chunk.get("parent_content")

                if score >= 0.8:
                    badge_color, badge_bg = "#4CAF50", "#1a3a1a"
                elif score >= 0.5:
                    badge_color, badge_bg = "#FF9800", "#3a2a1a"
                else:
                    badge_color, badge_bg = "#9e9e9e", "#1e1e1e"

                with st.expander(f"Chunk {i+1}  —  score {score:.4f}  —  `{chunk['chunk_id']}`", expanded=(i == 0)):
                    st.markdown(
                        f"""<div style='display:inline-block;padding:3px 10px;border-radius:20px;
                        background:{badge_bg};border:1px solid {badge_color};
                        color:{badge_color};font-size:13px;margin-bottom:8px'>
                        Score: {score:.4f}</div>""",
                        unsafe_allow_html=True,
                    )
                    if text.strip() == "[PARENT BLOCK RETRIEVED]" and parent:
                        st.markdown("**📄 Source content (parent block retrieved via graph traversal):**")
                        raw_text = parent.get("raw_text") or parent.get("content", "")
                        st.text_area("", value=raw_text, height=200, disabled=True, label_visibility="collapsed")
                        if st.checkbox("Show full parent JSON", key=f"pjson_{i}"):
                            st.json(parent)
                    else:
                        st.markdown("**📄 Chunk text:**")
                        st.text_area("", value=text, height=150, disabled=True, label_visibility="collapsed")
                        if parent:
                            if st.checkbox("Show parent block (graph traversal)", key=f"parent_{i}"):
                                raw = parent.get("raw_text") or parent.get("content", "")
                                if raw:
                                    st.text_area("", value=raw, height=150, disabled=True, label_visibility="collapsed")
                                else:
                                    st.json(parent)

        st.divider()
        st.caption("Query logged to `retrieval_audit_log`.")
        if st.button("View recent audit log"):
            try:
                df = query_db("SELECT * FROM retrieval_audit_log ORDER BY query_id DESC LIMIT 10")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load audit log: {e}")