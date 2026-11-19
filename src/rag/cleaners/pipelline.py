from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

from langchain_core.documents import Document

from .base import BaseCleaner
from .code_cleaner import CodeDocumentCleaner
from .docs_cleaner import DocumentCleaner
from .web_cleaner import WebDocumentCleaner


class CleanerPipeline(BaseCleaner):
    """Route every document to the cleaner that matches its origin/type."""

    _URL_PREFIXES: tuple[str, ...] = ("http://", "https://")
    _SUPPORTED_EXTENSIONS_CODE: frozenset[str] = frozenset({
        ".sql",
        ".cpp", ".cc", ".cxx", ".hpp", ".h",
        ".cs",
        ".go",
        ".html", ".htm",
        ".java",
        ".js", ".mjs", ".cjs", ".jsx",
        ".kt", ".kts",
        ".py",
        ".rs",
        ".ts", ".tsx",
    })
    _SUPPORTED_EXTENSIONS_DOCUMENT: frozenset[str] = frozenset({
        ".pdf",
        ".doc", ".docx", ".docs",
        ".pptx",
        ".md", ".markdown",
        ".txt",
    })
    _SUPPORTED_EXTENSIONS: frozenset[str] = (
        _SUPPORTED_EXTENSIONS_CODE | _SUPPORTED_EXTENSIONS_DOCUMENT
    )

    def clean(self, documents: list[Document]) -> list[Document]:
        cleaned: list[Document] = []
        for document in documents:
            cleaner = self._select_cleaner(document)
            cleaned.extend(cleaner.clean(documents=[document]))
        return cleaned

    def _select_cleaner(self, document: Document) -> BaseCleaner:
        metadata = document.metadata or {}
        source = str(metadata.get("source", ""))

        # Web pages first: a URL has no meaningful file extension.
        if self._is_url(source) or str(metadata.get("file_type", "")).lower() == "web":
            return WebDocumentCleaner()

        extension = self._get_extension(metadata)
        if extension in self._SUPPORTED_EXTENSIONS_DOCUMENT:
            return DocumentCleaner()
        if extension in self._SUPPORTED_EXTENSIONS_CODE:
            return CodeDocumentCleaner()
        raise ValueError(f"Unsupported document type: {extension!r} (source={source!r})")

    @classmethod
    def _get_extension(cls, metadata: dict) -> str:
        """Return a supported extension from ``file_type`` or ``source``."""
        file_type = str(metadata.get("file_type", "")).strip().lower()
        if file_type:
            if not file_type.startswith("."):
                file_type = f".{file_type}"
            if file_type in cls._SUPPORTED_EXTENSIONS:
                return file_type

        source = str(metadata.get("source", "")).strip()
        if source:
            # Support both Windows and POSIX style paths.
            extension = PureWindowsPath(source).suffix.lower() or PurePosixPath(source).suffix.lower()
            if extension in cls._SUPPORTED_EXTENSIONS:
                return extension

        raise ValueError(
            "Cannot determine a supported file extension from document metadata "
            f"(file_type={metadata.get('file_type')!r}, source={metadata.get('source')!r})."
        )

    def _is_url(self, source: object) -> bool:
        """Return whether ``source`` is an HTTP(S) URL string."""
        return isinstance(source, str) and source.lower().startswith(self._URL_PREFIXES)
