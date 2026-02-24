"""
KYC Document RAG — Streamlit Frontend
End-to-end pipeline: Upload -> S3 -> OCR -> Chunk -> Embed -> Retrieve
"""
import base64
import io
import time
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import os

# ── Page setup (must be the very first st.* call) ─────────────────────────────
st.set_page_config(
    page_title="KYC Document RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Deep-blue theme ───────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #07091a;
    color: #c8d6f0;
}
[data-testid="stSidebar"] {
    background-color: #0b0f25;
    border-right: 1px solid #1e2a4a;
}
[data-testid="stSidebar"] * { color: #9ab0d8 !important; }
h1 { color: #5ba3f5 !important; font-size: 1.45rem !important; letter-spacing:.02em; }
h2, h3 { color: #4d8fd4 !important; }
h4 { color: #7ab0e8 !important; }
[data-testid="stMetricValue"] { color: #5ba3f5 !important; }
[data-testid="stMetricLabel"] { color: #7a94c0 !important; }
input, textarea, select {
    background: #0f1530 !important;
    color: #c8d6f0 !important;
    border: 1px solid #1e2a4a !important;
}
[data-testid="stExpander"] {
    background: #0c1228 !important;
    border: 1px solid #1a2a50 !important;
    border-radius: 8px !important;
}
hr { border-color: #1e2a4a !important; }
[data-testid="stDataFrame"] { background: #0c1228 !important; }
small, [data-testid="stCaption"] { color: #6a84b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://api:8000/api/v1")
DB_URI = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'postgres')}"
    f"@{os.getenv('POSTGRES_SERVER', 'db')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'kyc_rag')}"
)
POLL_INTERVAL = 2
PREVIEW_W, PREVIEW_H = 640, 400  # fixed canvas for all preview types

@st.cache_resource
def get_engine():
    return create_engine(DB_URI, pool_pre_ping=True)

engine = get_engine()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### KYC RAG")
    st.caption("Document intelligence pipeline")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Ingest & Monitor", "Pipeline Explorer", "Retrieval"],
        label_visibility="collapsed",
        key="nav_page",
    )
    st.divider()
    st.caption("Services")
    try:
        requests.get(f"{API_URL.rsplit('/api', 1)[0]}/", timeout=2)
        st.markdown('<span style="color:#4fc86a;font-weight:600">API &nbsp; online</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span style="color:#d45a5a;font-weight:600">API &nbsp; offline</span>', unsafe_allow_html=True)


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
        ("upload",    "Upload to S3",       True,                              storage_uri or ""),
        ("ocr",       "OCR / Layout Parse", status.get("segments", 0) > 0,    f"{status.get('segments', 0)} segments"),
        ("chunking",  "Chunking",           status.get("chunks", 0) > 0,      f"{status.get('chunks', 0)} chunks"),
        ("nlp",       "Entity Extraction",  status.get("entities", 0) > 0,    f"{status.get('entities', 0)} entities"),
        ("embedding", "Embedding / Index",  status.get("embedded", 0) > 0,    f"{status.get('embedded', 0)} vectors"),
    ]

    cols = st.columns(len(stages))
    for col, (key, label, done, detail) in zip(cols, stages):
        with col:
            if done:
                bg, border, fg, sub = "#0d2a0d", "#2d6a2d", "#4fc86a", "#7ab87a"
            elif key == stage:
                bg, border, fg, sub = "#0a1035", "#2a4acc", "#5ba3f5", "#8ab0e0"
            else:
                bg, border, fg, sub = "#0c0f1e", "#1a2040", "#3a4a6a", "#2a3450"
            col.markdown(
                f"""<div style='text-align:center;padding:14px 8px;border-radius:8px;
                background:{bg};border:1px solid {border};margin:2px'>
                <div style='color:{fg};font-weight:700;font-size:13px;margin-bottom:4px'>{label}</div>
                <div style='color:{sub};font-size:11px'>{"DONE" if done else ("ACTIVE" if key == stage else "waiting")}</div>
                <div style='color:{sub};font-size:11px;margin-top:2px'>{detail if done else ""}</div>
                </div>""",
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
            st.markdown("#### Extracted Entities")
            for etype, group in df.groupby("entity_type"):
                values = ", ".join(f"`{v}`" for v in group["entity_value"].tolist())
                st.markdown(
                    f'<span style="color:#5ba3f5;font-weight:600">{etype}</span> — {values}',
                    unsafe_allow_html=True)
    except Exception:
        pass


def make_image_thumbnail(raw_bytes: bytes) -> bytes:
    """Resize to fit PREVIEW_W x PREVIEW_H canvas and centre on deep-blue background."""
    from PIL import Image
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
    canvas = Image.new("RGB", (PREVIEW_W, PREVIEW_H), (7, 9, 26))
    x = (PREVIEW_W - img.width) // 2
    y = (PREVIEW_H - img.height) // 2
    canvas.paste(img, (x, y))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def render_preview(fbytes: bytes, fname: str, ftype: str):
    """Always renders inside a fixed PREVIEW_W x PREVIEW_H box."""
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    st.markdown(
        f'<div style="color:#6a84b8;font-size:12px;margin-bottom:4px">Preview — {fname}</div>',
        unsafe_allow_html=True,
    )
    if ftype.startswith("image/") or ext in ("png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp"):
        try:
            thumb = make_image_thumbnail(fbytes)
            st.image(thumb, width=PREVIEW_W)
        except Exception:
            st.warning("Could not render image preview.")
    elif ftype == "application/pdf" or ext == "pdf":
        b64 = base64.b64encode(fbytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="{PREVIEW_W}px" height="{PREVIEW_H}px" '
            f'style="border:1px solid #1a2a50;border-radius:6px;display:block"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        size_kb = len(fbytes) / 1024
        st.markdown(
            f'<div style="width:{PREVIEW_W}px;height:{PREVIEW_H}px;display:flex;align-items:center;'
            f'justify-content:center;background:#0c1228;border:1px solid #1a2a50;border-radius:8px">'
            f'<div style="text-align:center;color:#4d6890">'
            f'<div style="font-size:42px;font-family:monospace;margin-bottom:12px">[{ext.upper() or "FILE"}]</div>'
            f'<div style="font-size:14px">{fname}</div>'
            f'<div style="font-size:12px;margin-top:4px">{size_kb:.1f} KB</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ── Status colour maps (no emoji) ────────────────────────────────────────────
_STATUS_BG = {"DONE": "#1a4d1a", "EMBEDDING": "#0e1e50", "NLP": "#0e2040",
               "CHUNKING": "#302800", "OCR": "#3a1010"}
_STATUS_FG = {"DONE": "#4fc86a", "EMBEDDING": "#5ba3f5", "NLP": "#4d8fd4",
               "CHUNKING": "#c8b84a", "OCR": "#d46060"}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INGEST & MONITOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "Ingest & Monitor":
    st.title("Ingest & Monitor")
    st.caption("Upload a KYC document and watch it move through every pipeline stage in real time.")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        # File picker outside form so preview renders immediately on selection
        uploaded_file = st.file_uploader(
            "Document (PDF, PNG, JPEG, TIFF)",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
        )
        if uploaded_file is not None:
            st.session_state["preview_bytes"] = uploaded_file.getvalue()
            st.session_state["preview_name"]  = uploaded_file.name
            st.session_state["preview_type"]  = uploaded_file.type or ""

        with st.form("ingest_form", clear_on_submit=False):
            customer_id = st.text_input("Customer ID", value="cust-001")
            submitted = st.form_submit_button("Upload & Ingest", use_container_width=True)

    with right_col:
        if "preview_bytes" in st.session_state:
            render_preview(
                st.session_state["preview_bytes"],
                st.session_state["preview_name"],
                st.session_state["preview_type"],
            )

    if submitted:
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            with st.spinner("Uploading to S3..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest",
                        data={"customer_id": customer_id},
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=30,
                    )
                    if resp.status_code == 202:
                        data = resp.json()
                        st.session_state["active_doc_id"]      = data["doc_id"]
                        st.session_state["active_task_id"]     = data["task_id"]
                        st.session_state["active_storage_uri"] = data["storage_uri"]
                        st.session_state["active_customer_id"] = customer_id
                        st.session_state["active_filename"]    = uploaded_file.name
                        st.session_state["poll_start"]         = time.time()
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
        st.markdown(
            f'<span style="color:#5ba3f5;font-weight:600;font-size:16px">{filename}</span>'
            f' &nbsp;<span style="color:#4d6890;font-size:13px">customer {cust}</span>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.caption(f"doc_id: {doc_id}")
        c2.caption(f"task_id: {task_id}")

        status  = pipeline_status(doc_id)
        is_done = status.get("embedded", 0) > 0

        render_pipeline_progress(status, storage_uri)

        status_slot = st.empty()  
        if not is_done:
            elapsed = int(time.time() - st.session_state.get("poll_start", time.time()))
            status_slot.markdown(
                f'<div style="background:#0a1030;border:1px solid #1a2a60;border-radius:6px;'
                f'padding:10px 16px;color:#5ba3f5;font-size:13px">'
                f'Processing... {elapsed}s elapsed — auto-refreshing every {POLL_INTERVAL}s</div>',
                unsafe_allow_html=True,
            )
            time.sleep(POLL_INTERVAL)
            status_slot.empty() 
            st.rerun()
        else:
            status_slot.markdown(
                '<div style="background:#0a1e0a;border:1px solid #2d6a2d;border-radius:6px;'
                'padding:10px 16px;color:#4fc86a;font-size:13px">'
                'Pipeline complete — document is ready for retrieval.</div>',
                unsafe_allow_html=True,
            )
            render_entities(doc_id)
            if st.button("Query this document"):
                st.session_state["prefill_customer_id"] = cust
                st.session_state["nav_page"] = "Retrieval"
                st.rerun()

    # ── Recent jobs table ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Recent Ingestions")
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
               FROM raw_document rd ORDER BY rd.doc_id DESC LIMIT 15"""
        )

        def _status_label(row):
            if row["embedded"] > 0: return "DONE"
            if row["entities"] > 0: return "EMBEDDING"
            if row["chunks"] > 0:   return "NLP"
            if row["segments"] > 0: return "CHUNKING"
            return "OCR"

        def _color_status(val):
            bg = _STATUS_BG.get(val, "#1a1a2a")
            fg = _STATUS_FG.get(val, "#9ab0d8")
            return f"background-color:{bg};color:{fg};font-weight:600;text-align:center"

        def _heatmap_col(s):
            mn, mx = s.min(), s.max()
            def _cell(v):
                if mx == mn or mx == 0:
                    return "background-color:#0c1228;color:#5ba3f5"
                ratio = (v - mn) / (mx - mn)
                g = int(10 + ratio * 45)
                b = int(40 + ratio * 90)
                return f"background-color:rgb(8,{g},{b});color:#a8d0ff"
            return s.map(_cell)

        recent["status"] = recent.apply(_status_label, axis=1)
        recent["storage_uri"] = recent["storage_uri"].str.replace(
            "s3://kyc-documents/", "s3://.../", regex=False
        )
        display = recent[["doc_id", "customer_id", "file_type", "status",
                           "segments", "chunks", "entities", "embedded", "storage_uri"]]
        styled = (
            display.style
            .applymap(_color_status, subset=["status"])
            .apply(_heatmap_col, subset=["segments", "chunks", "entities", "embedded"])
            .set_properties(**{"background-color": "#0c1228", "color": "#c8d6f0", "border-color": "#1a2a50"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load recent jobs: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PIPELINE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Pipeline Explorer":
    st.title("Pipeline Explorer")
    st.caption("Browse every layer of the pipeline database.")

    TABLE_META = {
        "raw_document":          {"desc": "One row per ingested document"},
        "parsed_layout_segment": {"desc": "OCR output — layout blocks per document"},
        "semantic_child_chunk":  {"desc": "Chunks with dense + sparse vectors"},
        "extracted_entity":      {"desc": "NER entities linked to chunks"},
        "knowledge_graph_edge":  {"desc": "Relationships between entity nodes"},
        "retrieval_audit_log":   {"desc": "Every retrieval request logged"},
    }

    st.markdown("#### Table Overview")
    cnt_cols = st.columns(len(TABLE_META))
    for col, (tbl, meta) in zip(cnt_cols, TABLE_META.items()):
        try:
            cnt = scalar_db(f"SELECT COUNT(*) FROM {tbl}")
        except Exception:
            cnt = "?"
        col.metric(tbl.replace("_", " ").title(), cnt)

    st.divider()

    col_left, col_right = st.columns([1, 3])
    with col_left:
        selected_table = st.radio(
            "Select table",
            list(TABLE_META.keys()),
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
elif page == "Retrieval":
    st.title("Retrieval")
    st.caption("Multi-layer gateway: Deterministic Bypass → Hybrid RRF Search → Cross-Encoder Reranking → Graph Traversal")

    default_cust = st.session_state.pop("prefill_customer_id", "")

    with st.form("retrieve_form"):
        query = st.text_input("Query", placeholder="Who is the passport holder and what is their nationality?")
        col1, col2 = st.columns([2, 1])
        with col1:
            customer_id = st.text_input("Customer ID (leave blank for all)", value=default_cust)
        with col2:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Retrieve", use_container_width=True)

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
                    badge_color, badge_bg = "#4fc86a", "#0a1e0a"
                elif score >= 0.5:
                    badge_color, badge_bg = "#f0a840", "#1e1200"
                else:
                    badge_color, badge_bg = "#7a94c0", "#0c1228"

                with st.expander(f"Chunk {i+1}  —  score {score:.4f}  —  {chunk['chunk_id']}", expanded=(i == 0)):
                    st.markdown(
                        f'<span style="background:{badge_bg};border:1px solid {badge_color};'
                        f'color:{badge_color};padding:2px 10px;border-radius:4px;font-size:13px">'
                        f'Score: {score:.4f}</span>',
                        unsafe_allow_html=True,
                    )
                    if text.strip() == "[PARENT BLOCK RETRIEVED]" and parent:
                        st.markdown('<span style="color:#5ba3f5;font-weight:600">Source content (parent block via graph traversal):</span>', unsafe_allow_html=True)
                        raw_text = parent.get("raw_text") or parent.get("content", "")
                        st.text_area("", value=raw_text, height=200, disabled=True, label_visibility="collapsed")
                        if st.checkbox("Show full parent JSON", key=f"pjson_{i}"):
                            st.json(parent)
                    else:
                        st.markdown('<span style="color:#5ba3f5;font-weight:600">Chunk text:</span>', unsafe_allow_html=True)
                        st.text_area("", value=text, height=150, disabled=True, label_visibility="collapsed")
                        if parent:
                            if st.checkbox("Show parent block (graph traversal)", key=f"parent_{i}"):
                                raw = parent.get("raw_text") or parent.get("content", "")
                                if raw:
                                    st.text_area("", value=raw, height=150, disabled=True, label_visibility="collapsed")
                                else:
                                    st.json(parent)

        st.divider()
        st.caption("Query logged to retrieval_audit_log.")
        if st.button("View recent audit log"):
            try:
                df = query_db("SELECT * FROM retrieval_audit_log ORDER BY query_id DESC LIMIT 10")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load audit log: {e}")