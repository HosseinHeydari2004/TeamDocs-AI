from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker


class TXTChunker(BaseChunker):
    def __init(self, chunk_size: int = 1200, overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk(self, docs: list[Document]) -> list[Document]:
        chunks = []
        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue
            text_document = Document(
                page_content=text,
                metadata={
                    **doc.metadata,
                    "element_type": "text"
                }
            )
            document_chunk = self._splitter.split_documents([text_document])
            chunks.extend(document_chunk)
        return chunks
