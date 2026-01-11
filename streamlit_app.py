import os
import time
import tempfile
import streamlit as st
from collections import Counter

from ingestion import ingest_document
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from rag.retriever import Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.qa_chain import answer_question, summarize_answer

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal RAG Chat + Evaluation",
    layout="wide",
)

st.title("💬 Multi-Modal Document Chat (Hybrid RAG)")
st.caption(
    "Chat with documents containing text, tables, and images (OCR) "
    "with built-in retrieval & latency evaluation."
)

# ------------------------------------------------
# Session state initialization
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

if "summary_added" not in st.session_state:
    st.session_state.summary_added = False

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

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
            st.session_state.last_summary = None
            st.session_state.summary_added = False
            st.session_state.last_metrics = None

        st.success("✅ Document indexed with Hybrid Retrieval")

    show_sources = st.checkbox("Show retrieved context")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_answer = None
        st.session_state.last_summary = None
        st.session_state.summary_added = False
        st.session_state.last_metrics = None

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
# Chat input & QA
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

        # ---- Retrieval (with metrics) ----
        retrieval_start = time.time()
        retrieved_chunks = st.session_state.retriever.retrieve(user_input)
        retrieval_latency = round(time.time() - retrieval_start, 3)

        # ---- Retrieval metrics ----
        modalities = Counter(c.modality for c in retrieved_chunks)
        unique_pages = len(set(c.page for c in retrieved_chunks))
        avg_chunk_len = round(
            sum(len(c.content) for c in retrieved_chunks) / max(len(retrieved_chunks), 1),
            1,
        )

        # ---- Generation (with metrics) ----
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                gen_start = time.time()
                answer = answer_question(retrieved_chunks, user_input)
                gen_latency = round(time.time() - gen_start, 3)

                st.markdown(answer)

                st.session_state.last_answer = answer
                st.session_state.last_summary = None
                st.session_state.summary_added = False

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

        # ---- Save metrics ----
        st.session_state.last_metrics = {
            "retrieval_latency_sec": retrieval_latency,
            "generation_latency_sec": gen_latency,
            "num_chunks": len(retrieved_chunks),
            "unique_pages": unique_pages,
            "avg_chunk_length": avg_chunk_len,
            "modalities": dict(modalities),
            "answer_tokens": len(answer.split()),
        }

# ------------------------------------------------
# Evaluation Dashboard (Built-in)
# ------------------------------------------------
if st.session_state.last_metrics:
    st.markdown("---")
    st.subheader("📊 Evaluation Dashboard")

    m = st.session_state.last_metrics

    col1, col2 = st.columns(2)
    col1.metric("Retrieval Latency (sec)", m["retrieval_latency_sec"])
    col2.metric("Generation Latency (sec)", m["generation_latency_sec"])

    col3, col4, col5 = st.columns(3)
    col3.metric("Chunks Retrieved", m["num_chunks"])
    col4.metric("Unique Pages", m["unique_pages"])
    col5.metric("Avg Chunk Length", m["avg_chunk_length"])

    st.subheader("🧩 Modality Distribution")
    st.json(m["modalities"])

    st.subheader("📝 Answer Metrics")
    st.metric("Answer Token Count", m["answer_tokens"])
