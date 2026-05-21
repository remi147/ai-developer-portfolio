"""Streamlit UI for the RAG Document Q&A System."""
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from ingest import ingest_file, INDEX_PATH
from qa_chain import build_qa_chain

load_dotenv()

st.set_page_config(page_title="📄 RAG Q&A", page_icon="🤖", layout="wide")
st.title("🤖 RAG Document Q&A")
st.caption("Upload a PDF or text file, then ask questions about its contents.")

# ── Sidebar: document upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("Choose a PDF or .txt file", type=["pdf", "txt"])
    if uploaded and st.button("Ingest document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        with st.spinner("Chunking and embedding…"):
            n = ingest_file(tmp_path)
        st.success(f"Ingested {n} chunks!")

# ── Main: Q&A ─────────────────────────────────────────────────────────────────
query = st.text_input("💬 Ask a question about your document:")

if query:
    if not Path(INDEX_PATH).exists():
        st.warning("No documents ingested yet. Please upload a file first.")
    else:
        with st.spinner("Retrieving and generating answer…"):
            qa = build_qa_chain()
            result = qa.invoke({"query": query})
        st.markdown("### Answer")
        st.write(result["result"])
        if result.get("source_documents"):
            with st.expander("📚 Source chunks"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(doc.page_content[:500])
