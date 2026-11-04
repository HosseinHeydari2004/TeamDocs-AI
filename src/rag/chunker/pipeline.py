from pathlib import Path
from urllib.parse import urlparse

from langchain_core.documents import Document

from .base import BaseChunker
from .code import CodeChunker
from .markdown import MarkdownChunker
from .pdf import PDFChunker
from .text import TXTChunker
from .web import WebChunker


class ChunkerPipeline:
    """Routes documents to the appropriate chunking strategy."""

    CODE_EXTENSIONS = {
        ".py", ".js", ".jsx", ".ts", ".tsx",
        ".java", ".cpp", ".cc", ".cxx", ".c",
        ".h", ".hpp", ".go", ".rs", ".php",
        ".rb", ".swift", ".kt", ".scala",
        ".sql", ".ddl", ".psql", ".pgsql", ".mysql",
    }

    def __init__(
            self,
            chunk_size: int = 1200,
            overlap: int = 200,
    ) -> None:
        self.chunkers: dict[str, BaseChunker] = {
            "code": CodeChunker(chunk_size, overlap),
            "markdown": MarkdownChunker(),
            "pdf": PDFChunker(chunk_size, overlap),
            "text": TXTChunker(chunk_size, overlap),
            "web": WebChunker(chunk_size, overlap),
        }

    def _detect_type(self, doc: Document) -> str:
        metadata = doc.metadata or {}

        # Explicit document type takes priority.
        doc_type = metadata.get("document_type")
        if doc_type in self.chunkers:
            return doc_type

        source = str(metadata.get("source", ""))
        source_type = str(metadata.get("source_type", "")).lower()

        # Web loaders should set source_type="web".
        if source_type in {"web", "url"}:
            return "web"

        if source.startswith(("http://", "https://")):
            return "web"

        suffix = Path(urlparse(source).path).suffix.lower()

        if suffix == ".pdf":
            return "pdf"

        if suffix in {".md", ".markdown", ".mdx"}:
            return "markdown"

        if suffix in self.CODE_EXTENSIONS:
            return "code"

        # Default for plain text and unknown extensions.
        return "text"

    def chunk(
            self,
            docs: list[Document],
    ) -> list[Document]:
        chunks: list[Document] = []

        for doc in docs:
            if not doc.page_content or not doc.page_content.strip():
                continue

            doc_type = self._detect_type(doc)
            chunker = self.chunkers[doc_type]

            document_chunks = chunker.chunk([doc])

            for chunk_index, chunk in enumerate(document_chunks):
                metadata = {**doc.metadata, **chunk.metadata, "document_type": doc_type, "chunk_index": chunk_index}

                chunks.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=metadata,
                    )
                )

        return chunks
