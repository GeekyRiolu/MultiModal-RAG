from typing import List
from utils.schemas import DocumentChunk
import uuid

def chunk_text(text: str, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


def chunk_documents(docs: List[DocumentChunk]) -> List[DocumentChunk]:
    final_chunks = []

    for doc in docs:
        sub_chunks = chunk_text(doc.content)

        for sub in sub_chunks:
            final_chunks.append(
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    modality=doc.modality,
                    content=sub,
                    page=doc.page
                )
            )

    return final_chunks
