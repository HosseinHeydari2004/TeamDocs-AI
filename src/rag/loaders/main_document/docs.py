from pathlib import Path

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseLoader


class DocsLoader(BaseLoader):
    """Load .docx files (legacy .doc requires LibreOffice on the host)."""

    def load(self, path: Path | str) -> list[Document]:
        loader = UnstructuredWordDocumentLoader(
            file_path=str(path),
            mode="single",
        )
        return loader.load()
