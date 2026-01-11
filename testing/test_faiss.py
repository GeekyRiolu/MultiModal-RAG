from ingestion import ingest_document

from chunking.chunker import chunk_documents

from embeddings.embedder import Embedder

from vectorstore.faiss_store import FAISSStore

docs = ingest_document("data/raw_docs/qatar_test_doc.pdf")

chunks = chunk_documents(docs)

embedder = Embedder()

embeddings = embedder.embed([c.content for c in chunks])

store = FAISSStore(dim=len(embeddings[0]))

store.add(embeddings, chunks)

print("FAISS index size:", store.index.ntotal)