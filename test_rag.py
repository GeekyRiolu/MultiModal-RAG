# from ingestion import ingest_document
# from chunking.chunker import chunk_documents
# from embeddings.embedder import Embedder
# from vectorstore.faiss_store import FAISSStore
# from rag.retriever import Retriever
# from rag.qa_chain import answer_question

# docs = ingest_document("data/raw_docs/qatar_test_doc.pdf")
# chunks = chunk_documents(docs)

# embedder = Embedder()
# embeddings = embedder.embed([c.content for c in chunks])

# store = FAISSStore(dim=len(embeddings[0]))
# store.add(embeddings, chunks)

# retriever = Retriever(store, embedder)

# question = "What are the main risks listed in the Risk Assessment Matrix?"
# retrieved_chunks = retriever.retrieve(question)

# answer = answer_question(retrieved_chunks, question)
# print(answer)


from ingestion import ingest_document
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from rag.retriever import Retriever
from rag.qa_chain import answer_question


PDF_PATH = "data/raw_docs/qatar_test_doc.pdf"


def main():
    print("\n=== IMAGE / OCR RAG TEST ===\n")

    # 1️⃣ Ingest document
    docs = ingest_document(PDF_PATH)
    chunks = chunk_documents(docs)

    # Debug: count image chunks
    image_chunks = [c for c in chunks if c.modality == "image"]
    print(f"Total chunks: {len(chunks)}")
    print(f"Image/OCR chunks: {len(image_chunks)}")

    if not image_chunks:
        print("❌ No image chunks found. OCR pipeline may be missing.")
        return

    # Show sample OCR content
    print("\n--- Sample OCR Content ---")
    for c in image_chunks[:2]:
        print(f"\nPage {c.page}")
        print(c.content[:300])
        print("-" * 50)

    # 2️⃣ Build embeddings + FAISS
    embedder = Embedder()
    embeddings = embedder.embed([c.content for c in chunks])

    store = FAISSStore(dim=len(embeddings[0]))
    store.add(embeddings, chunks)

    retriever = Retriever(store, embedder, top_k=5)

    # 3️⃣ Image-focused question (IMPORTANT)
    question = (
        "According to the chart or figure in the document, "
        "what factors contribute to downside or global risks?"
    )

    print("\nQuestion:")
    print(question)

    # 4️⃣ Retrieve relevant chunks
    retrieved_chunks = retriever.retrieve(question)

    print("\n--- Retrieved Chunks ---")
    for c in retrieved_chunks:
        print(f"Page {c.page} | Modality: {c.modality}")
        print(c.content[:200])
        print("-" * 50)

    # 5️⃣ Generate answer using Gemini RAG
    answer = answer_question(retrieved_chunks, question)

    print("\n=== GEMINI ANSWER ===")
    print(answer)
    print("\n=== END IMAGE / OCR TEST ===\n")


if __name__ == "__main__":
    main()
