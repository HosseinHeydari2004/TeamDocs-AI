from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseChunker


class PDFChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, doc: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for page_doc in doc:
            text = page_doc.page_content.strip()
            if not text:
                continue
            page_metadata = page_doc.metadata
            page_metadata.setdefault("element_type", "text")

            page_document = Document(
                page_content=text,
                metadata=page_metadata,
            )
            page_chunks = self._splitter.split_documents([page_document])
            for index, chunk in enumerate(page_chunks):
                chunk.metadata["chunk_index"] = index

            chunks.extend(page_chunks)
        return chunks
