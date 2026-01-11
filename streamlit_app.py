import streamlit as st
import tempfile
import os

from ingestion import ingest_document
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from rag.retriever import Retriever
from rag.qa_chain import answer_question

st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")

st.title("📄 Multi-Modal Document QA (RAG)")
st.write(
    "Ask questions over documents containing **text, tables, and images (OCR)**."
)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.embedder = None
    st.session_state.chunks = None

# ---- Upload PDF ----
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Ingestion + chunking
        docs = ingest_document(pdf_path)
        chunks = chunk_documents(docs)

        # Embeddings + FAISS
        embedder = Embedder()
        embeddings = embedder.embed([c.content for c in chunks])

        store = FAISSStore(dim=len(embeddings[0]))
        store.add(embeddings, chunks)

        retriever = Retriever(store, embedder)

        # Save in session
        st.session_state.vector_store = store
        st.session_state.retriever = retriever
        st.session_state.embedder = embedder
        st.session_state.chunks = chunks

    st.success("✅ Document processed successfully!")

# ---- Question Answering ----
if st.session_state.retriever:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Retrieving and generating answer..."):
            retrieved_chunks = st.session_state.retriever.retrieve(question)
            answer = answer_question(retrieved_chunks, question)

        st.subheader("📌 Answer")
        st.write(answer)

        # Optional: show retrieved sources
        with st.expander("🔍 Retrieved Context (for transparency)"):
            for c in retrieved_chunks:
                st.markdown(
                    f"**Page {c.page} | {c.modality.upper()}**\n\n{c.content[:500]}"
                )
