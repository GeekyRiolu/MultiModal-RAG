import sys
import os
import tempfile
import streamlit as st

from ingestion import ingest_document
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from rag.retriever import Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.qa_chain import answer_question

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal RAG Chat",
    layout="wide",
)

st.title("💬 Multi-Modal Document Chat")
st.caption(
    "Chat with documents containing **text, tables, charts, and images (OCR)** "
    "using **Hybrid Retrieval**."
)

# ------------------------------------------------
# Session State Initialization
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

# ------------------------------------------------
# Sidebar — Upload & Controls
# ------------------------------------------------
with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            # ---- Ingestion + Chunking ----
            docs = ingest_document(pdf_path)
            chunks = chunk_documents(docs)

            # ---- Embeddings + FAISS ----
            embedder = Embedder()
            embeddings = embedder.embed([c.content for c in chunks])

            store = FAISSStore(dim=len(embeddings[0]))
            store.add(embeddings, chunks)

            # ---- Hybrid Retriever ----
            dense_retriever = Retriever(store, embedder)
            hybrid_retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                chunks=chunks,
                top_k=5
            )

            # ---- Save to session ----
            st.session_state.retriever = hybrid_retriever
            st.session_state.chunks = chunks
            st.session_state.messages = []

        st.success("✅ Document indexed with Hybrid Retrieval")

    show_sources = st.checkbox("Show retrieved context")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

# ------------------------------------------------
# Chat History
# ------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------
# Chat Input
# ------------------------------------------------
if st.session_state.retriever:
    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        # ---- User message ----
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # ---- Assistant message ----
        with st.chat_message("assistant"):
            with st.spinner("Retrieving with Hybrid RAG..."):
                retrieved_chunks = st.session_state.retriever.retrieve(user_input)
                answer = answer_question(retrieved_chunks, user_input)

                st.markdown(answer)

                if show_sources:
                    st.markdown("---")
                    st.markdown("**🔍 Retrieved Context**")
                    for c in retrieved_chunks:
                        st.markdown(
                            f"- **Page {c.page} | {c.modality.upper()}**: "
                            f"{c.content[:200]}..."
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("⬅️ Upload a PDF from the sidebar to start chatting.")
