import camelot
from utils.schemas import DocumentChunk
import uuid


def extract_tables(pdf_path: str):
    tables = camelot.read_pdf(pdf_path, pages="all")
    chunks = []

    for table in tables:
        table_text = table.df.to_string(index=False)

        chunks.append(
            DocumentChunk(
                id=str(uuid.uuid4()),
                modality="table",
                content=table_text,
                page=table.page
            )
        )

    return chunks
