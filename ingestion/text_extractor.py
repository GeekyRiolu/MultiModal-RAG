import fitz  # PyMuPDF
from utils.schemas import DocumentChunk
import uuid


def extract_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue

        chunks.append(
            DocumentChunk(
                id=str(uuid.uuid4()),
                modality="text",
                content=text,
                page=page_num
            )
        )

    return chunks
