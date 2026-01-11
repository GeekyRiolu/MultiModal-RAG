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
from rag.qa_chain import (
    answer_question,
    summarize_answer,
)

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal RAG Chat",
    layout="wide",
)

st.title("💬 Multi-Modal Document Chat (Hybrid RAG)")
st.caption(
    "Ask detailed questions over documents containing text, tables, and images (OCR)."
)

# ------------------------------------------------
# Session state
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
    
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
# ------------------------------------------------
# Sidebar — Upload
# ------------------------------------------------
with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            docs = ingest_document(pdf_path)
            chunks = chunk_documents(docs)

            embedder = Embedder()
            embeddings = embedder.embed([c.content for c in chunks])

            store = FAISSStore(dim=len(embeddings[0]))
            store.add(embeddings, chunks)

            dense = Retriever(store, embedder)
            retriever = HybridRetriever(
                dense_retriever=dense,
                chunks=chunks,
                top_k=5,
            )

            st.session_state.retriever = retriever
            st.session_state.chunks = chunks
            st.session_state.messages = []
            st.session_state.last_answer = None

        st.success("✅ Document indexed")

    show_sources = st.checkbox("Show retrieved context")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_answer = None

    if st.session_state.messages:
        chat_md = "\n\n".join(
            f"**{m['role'].upper()}**:\n{m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Export Chat",
            chat_md,
            file_name="chat_history.md",
        )

# ------------------------------------------------
# Chat history
# ------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------
# Chat input
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
            with st.spinner("Retrieving and generating answer..."):
                retrieved_chunks = st.session_state.retriever.retrieve(user_input)
                answer = answer_question(retrieved_chunks, user_input)

                st.markdown(answer)
                st.session_state.last_answer = answer

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

# ------------------------------------------------
# Summarize last answer (NO re-retrieval)
# ------------------------------------------------
if st.session_state.last_answer:
    st.markdown("---")

    if st.button("✂️ Summarize last answer"):
        with st.spinner("Summarizing answer..."):
            st.session_state.last_summary = summarize_answer(
                st.session_state.last_answer
            )

if st.session_state.last_summary:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.last_summary)

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state.last_summary}
    )

