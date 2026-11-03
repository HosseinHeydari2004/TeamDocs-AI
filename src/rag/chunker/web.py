
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker


class WebChunker(BaseChunker):
    """Chunker for web page content."""

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                "",
            ],
            add_start_index=True,
        )

    def chunk(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        for doc in docs:
            text = doc.page_content.strip()

            if not text:
                continue

            metadata = dict(doc.metadata)
            metadata.setdefault("element_type", "web_content")

            web_document = Document(
                page_content=text,
                metadata=metadata,
            )

            chunks.extend(
                self._splitter.split_documents([web_document])
            )

        return chunks
