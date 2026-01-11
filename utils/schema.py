from dataclasses import dataclass

@dataclass
class DocumentChunk:
    id: str
    modality: str      # "text" | "table" | "image"
    content: str
    page: int
