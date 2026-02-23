import streamlit as st
import requests
import pandas as pd
from sqlalchemy import create_engine, text
import os
import json
import uuid

API_URL = os.getenv("API_URL", "http://api:8000/api/v1")
DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_SERVER')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

engine = create_engine(DB_URI)

st.set_page_config(page_title="KYC RAG Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ingestion", "Pipeline & DB Explorer", "Retrieval"])

if page == "Ingestion":
    st.title("Document Ingestion")
    st.write("Upload multiple documents to trigger ingestion jobs. The files will be processed by the Celery workers.")
    
    customer_id = st.text_input("Customer ID", value="cust_123")
    metadata_str = st.text_area("Metadata (JSON)", value='{"force_ocr": true}')
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    
    if st.button("Ingest Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            for file in uploaded_files:
                # Save to shared volume
                file_path = f"/shared/{uuid.uuid4()}_{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Determine file type
                ext = file.name.split('.')[-1].upper()
                if ext == 'JPG': ext = 'JPEG'
                
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata.")
                    break
                
                payload = {
                    "customer_id": customer_id,
                    "file_type": ext,
                    "storage_uri": f"file://{file_path}",
                    "metadata": metadata
                }
                
                try:
                    resp = requests.post(f"{API_URL}/ingest", json=payload)
                    if resp.status_code == 202:
                        st.success(f"Started ingestion for {file.name}. Task ID: {resp.json()['task_id']}")
                    else:
                        st.error(f"Failed to ingest {file.name}: {resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

elif page == "Pipeline & DB Explorer":
    st.title("Pipeline & Database Explorer")
    st.write("View how documents are chunked, transformed, and stored in the database.")
    
    tables = [
        "raw_document", 
        "parsed_layout_segment", 
        "semantic_child_chunk", 
        "extracted_entity", 
        "knowledge_graph_edge", 
        "retrieval_audit_log"
    ]
    selected_table = st.selectbox("Select Table", tables)
    
    if st.button("Query Table"):
        try:
            with engine.connect() as conn:
                if selected_table == "semantic_child_chunk":
                    # Exclude dense_vector to avoid massive output
                    query = text(f"SELECT chunk_id, segment_id, text_content, sparse_vector FROM {selected_table} LIMIT 100")
                else:
                    query = text(f"SELECT * FROM {selected_table} LIMIT 100")
                
                df = pd.read_sql(query, conn)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Database query failed: {e}")

elif page == "Retrieval":
    st.title("Retrieval & Querying")
    st.write("Test the multi-layered retrieval gateway (Deterministic Bypass, Hybrid Search, RRF, Cross-Encoder Reranking).")
    
    query = st.text_input("Query", value="Who is the UBO and what is their Passport A1234?")
    customer_id = st.text_input("Customer ID (Optional)", value="")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
    
    if st.button("Retrieve"):
        payload = {
            "query": query,
            "top_k": top_k
        }
        if customer_id:
            payload["customer_id"] = customer_id
            
        try:
            resp = requests.post(f"{API_URL}/retrieve", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                st.subheader("Retrieval Result")
                st.write(f"**Router Decision:** {data['router_decision']}")
                st.write(f"**Confidence Scores:** {data['confidence_scores']}")
                
                st.subheader("Retrieved Context")
                for idx, chunk in enumerate(data['retrieved_chunks']):
                    with st.expander(f"Chunk {idx+1} (Score: {chunk['score']:.4f})"):
                        st.write("**Text Content:**")
                        st.write(chunk['text_content'])
                        if chunk.get('parent_content'):
                            st.write("**Parent Content (Graph Traversal):**")
                            st.json(chunk['parent_content'])
            else:
                st.error(f"Retrieval failed: {resp.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")