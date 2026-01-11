import fitz
import pytesseract
from PIL import Image
import io
import uuid
from utils.schemas import DocumentChunk


def extract_images_ocr(pdf_path: str):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image).strip()

            if ocr_text:
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        modality="image",
                        content=ocr_text,
                        page=page_index + 1
                    )
                )

    return chunks
