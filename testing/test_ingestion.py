from ingestion import ingest_document

chunks = ingest_document("data/raw_docs/qatar_test_doc.pdf")

for c in chunks[:5]:
    print(c.modality, c.page)
    print(c.content[:200])
    print("-" * 50)

print(f"Total chunks: {len(chunks)}")
