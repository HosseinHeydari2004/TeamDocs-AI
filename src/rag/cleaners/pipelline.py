from __future__ import annotations
from .base import BaseCleaner
from langchain_core.documents import Document
from .web_cleaner import WebDocumentCleaner
from .docs_cleaner import DocumentCleaner
from .code_cleaner import CodeDocumentCleaner
from typing import Any
from pathlib import PurePosixPath, PureWindowsPath


class CleanerPipeline(BaseCleaner):
    _URL_PREFIXES: tuple[str, ...] = ("http://", "https://")
    _SUPPORTED_EXTENSIONS_CODE = {
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

    }
    _SUPPORTED_EXTENSIONS_DOCUMENT: dict[str, Any] = {
        ".pdf",
        ".doc",
        ".docx",
        ".docs",
        ".md",
        ".markdown",
        ".txt",
    }
    _SUPPORTED_EXTENSIONS: set[str] = (
            _SUPPORTED_EXTENSIONS_CODE
            | _SUPPORTED_EXTENSIONS_DOCUMENT
    )

    def clean(self, documents: list[Document]) -> list[Document]:
        docs:list[Document] = []
        for document in documents:
            extension = self._get_extension(metadata=document.metadata.copy())
            if extension in self._SUPPORTED_EXTENSIONS_DOCUMENT:
                cleaner = DocumentCleaner()
            elif extension in self._SUPPORTED_EXTENSIONS_CODE:
                cleaner = CodeDocumentCleaner()
            elif self._is_url(source=document.metadata["source"]):
                cleaner = WebDocumentCleaner()
            else:
                raise ValueError(f"Unsupported document type {document.metadata.type}")
            docs.extend(cleaner.clean(documents=[document]))
        return docs


    @classmethod
    def _get_extension(cls, metadata: dict) -> str:
        """Get a supported extension from metadata."""

        file_type = str(metadata.get("file_type", "")).strip().lower()

        if file_type:
            if not file_type.startswith("."):
                file_type = f".{file_type}"

            if file_type in cls._SUPPORTED_EXTENSIONS:
                return file_type

        source = str(metadata.get("source", "")).strip()

        if source:
            # Support both Windows and Unix-style paths.
            extension = PureWindowsPath(source).suffix.lower()

            if not extension:
                extension = PurePosixPath(source).suffix.lower()

            if extension in cls._SUPPORTED_EXTENSIONS:
                return extension

        raise ValueError(
            "Cannot determine supported code file extension "
            "from document metadata."
        )

    def _is_url(self, source:object| str) -> bool:
        """
        Return whether ``source`` is an HTTP or HTTPS URL string.

        Parameters
        ----------
        source : object
            Candidate source; non-string values always return ``False``.

        Returns
        -------
        bool
            ``True`` if ``source`` is a string starting with a supported URL
            prefix, otherwise ``False``.
        """
        return isinstance(source, str) and source.lower().startswith(self._URL_PREFIXES)
