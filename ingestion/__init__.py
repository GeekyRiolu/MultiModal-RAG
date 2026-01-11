from ingestion.text_extractor import extract_text
from ingestion.table_extractor import extract_tables
from ingestion.image_extractor import extract_images_ocr


def ingest_document(pdf_path: str):
    text_chunks = extract_text(pdf_path)
    table_chunks = extract_tables(pdf_path)
    image_chunks = extract_images_ocr(pdf_path)

    return text_chunks + table_chunks + image_chunks
