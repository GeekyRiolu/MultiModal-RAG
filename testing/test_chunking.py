from ingestion import ingest_document
from chunking.chunker import chunk_documents

docs = ingest_document("data/raw_docs/qatar_test_doc.pdf")
chunks = chunk_documents(docs)

print(f"Before chunking: {len(docs)}")
print(f"After chunking: {len(chunks)}")

print(chunks[0].content[:300])
