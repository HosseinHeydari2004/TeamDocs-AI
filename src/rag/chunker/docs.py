from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker


class DocsChunker(BaseChunker):

    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self._chunk_size = chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk(self, doc: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        buffer: list[Document] = []
        buffer_size: int = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_size
            if not buffer:
                return

            text = "\n\n".join(
                item.page_content.strip()
                for item in buffer
                if item.page_content.strip()
            )
            if not text:
                buffer = []
                buffer_size = 0
                return

            metadata = dict(buffer[0].metadata)
            metadata["element_type"] = "text"
            merged_doc = Document(
                page_content=text,
                metadata=metadata,
            )
            chunks.extend(self._splitter.split_documents([merged_doc]))
            buffer = []
            buffer_size = 0

        for element in doc:
            text = element.page_content.strip()
            if not text:
                continue

            metadata = dict(element.metadata)
            category = metadata.get("category", "")

            if category == "Table":
                flush_buffer()
                table_doc = Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "element_type": "table",
                    },
                )
                chunks.extend(self._splitter.split_documents([table_doc]))
                continue
            elif category == "Title":
                flush_buffer()

            if buffer and buffer_size + len(text) > self._chunk_size:
                flush_buffer()

            buffer.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )
            buffer_size += len(text)

        flush_buffer()
        return chunks
