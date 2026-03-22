import streamlit as st
import os
import time

st.set_page_config(page_title="Finance Document Intelligence Copilot", page_icon="📊", layout="wide")

from src.rag_pipeline import index_documents, query_documents
from src.mongodb_handler import get_stats
from config import DOCS_FOLDER

st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: bold; color: #1f4e79; }
.answer-box { background: #f0f7ff; border-left: 4px solid #1f4e79; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
.badge { display: inline-block; background: #e8f4ea; color: #2d6a35; border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.75rem; font-weight: bold; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Finance Document Intelligence Copilot</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#666;">RAG-powered Q&A · Hugging Face · FAISS · MongoDB · Streamlit</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Document Management")
    st.markdown("---")
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload financial PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join(DOCS_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")

    existing = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")] if os.path.exists(DOCS_FOLDER) else []
    if existing:
        st.markdown("**Available documents:**")
        for pdf in existing:
            st.markdown(f"• `{pdf}`")
    else:
        st.info("No PDFs yet. Upload some above.")

    st.markdown("---")
    st.subheader("2. Index Documents")
    if st.button("⚡ Index Documents", type="primary", use_container_width=True):
        with st.spinner("Indexing... (1-3 mins for large PDFs)"):
            start = time.time()
            success = index_documents(DOCS_FOLDER)
            elapsed = time.time() - start
        if success:
            st.success(f"✅ Done in {elapsed:.1f}s")
            st.balloons()
        else:
            st.error("❌ No PDFs found. Please upload documents first.")

    st.markdown("---")
    st.subheader("📊 Index Status")
    try:
        stats = get_stats()
        st.metric("Chunks indexed", stats["total_chunks"])
        st.metric("Documents", len(stats["documents"]))
        if stats["documents"]:
            st.markdown("**Indexed files:**")
            for doc in stats["documents"]:
                st.markdown(f"• `{doc}`")
    except:
        st.warning("MongoDB not connected or empty.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Question")
    examples = ["", "What was the total revenue?", "What are the main risk factors?", "What is the company's cash position?", "What were the operating expenses?"]
    selected = st.selectbox("Try an example:", examples)
    user_question = st.text_area("Your question:", value=selected, height=80, placeholder="e.g. What was net income last year?", label_visibility="collapsed")
    ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

with col2:
    st.subheader("⚙️ Settings")
    show_scores = st.checkbox("Show relevance scores", value=True)
    show_raw = st.checkbox("Show raw chunk text", value=False)

if ask_btn and user_question.strip():
    st.markdown("---")
    with st.spinner("🔍 Searching and generating answer..."):
        result = query_documents(user_question)

    st.subheader("🤖 Answer")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    sources = result.get("sources", [])
    if sources:
        st.subheader(f"📎 Sources ({len(sources)})")
        for i, src in enumerate(sources, 1):
            label = f"Source {i} · {src['filename']} · Page {src['page_num']}"
            if show_scores:
                label += f" · Score: {src['relevance_score']:.2%}"
            with st.expander(label, expanded=(i == 1)):
                if show_raw:
                    st.code(src["text"])
                else:
                    st.markdown(src["text"])
elif ask_btn:
    st.warning("Please enter a question.")
else:
    st.markdown("---")
    st.subheader("🚀 How to get started")
    st.markdown("""
1. **Upload PDFs** in the sidebar (annual reports, earnings calls, etc.)
2. Click **⚡ Index Documents** to process them
3. **Type your question** and click **🔍 Ask**
    """)
    with st.expander("🏗️ How the RAG system works"):
        st.code("""
Your PDF
  ↓ PyMuPDF — extract text
Text Chunks (500 chars each)
  ↓ Hugging Face — all-MiniLM-L6-v2
Embeddings (384-dim vectors)
  ↓                 ↓
MongoDB           FAISS Index
(stores text)     (stores vectors)

── At Query Time ──
Your Question → Embed → FAISS Search
→ Fetch from MongoDB → LLM → Answer ✓
        """)
