from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    """Load Markdown as raw text so headings remain available to the chunker."""

    def load(self, path: Path | str) -> list[Document]:
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
