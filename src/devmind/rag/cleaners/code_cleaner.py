import re
from pathlib import PurePosixPath, PureWindowsPath

from langchain_core.documents import Document

from .base import BaseCleaner


class CodeDocumentCleaner(BaseCleaner):
    """Clean source-code documents while preserving code structure."""

    SUPPORTED_EXTENSIONS = {
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

    def clean(self, documents: list[Document]) -> list[Document]:
        cleaned_documents = []

        for document in documents:
            if not isinstance(document.page_content, str):
                continue

            text = document.page_content
            metadata = document.metadata.copy()

            extension = self._get_extension(metadata)

            if not text.strip():
                continue

            text = self._normalize_line_endings(text)
            text = self._remove_null_bytes(text)
            text = self._remove_unwanted_control_characters(text)

            if extension in {".html", ".htm"}:
                text = self._clean_html(text)
            else:
                text = self._clean_source_code(text)

            if not text.strip():
                continue

            metadata["file_type"] = extension.lstrip(".")
            metadata["content_type"] = "source_code"

            cleaned_documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return cleaned_documents

    @classmethod
    def _get_extension(cls, metadata: dict) -> str:
        """Get a supported extension from metadata."""

        file_type = str(metadata.get("file_type", "")).strip().lower()

        if file_type:
            if not file_type.startswith("."):
                file_type = f".{file_type}"

            if file_type in cls.SUPPORTED_EXTENSIONS:
                return file_type

        source = str(metadata.get("source", "")).strip()

        if source:
            # Support both Windows and Unix-style paths.
            extension = PureWindowsPath(source).suffix.lower()

            if not extension:
                extension = PurePosixPath(source).suffix.lower()

            if extension in cls.SUPPORTED_EXTENSIONS:
                return extension

        raise ValueError(
            "Cannot determine supported code file extension "
            "from document metadata."
        )

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        """Normalize line endings without changing indentation."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _remove_null_bytes(text: str) -> str:
        return text.replace("\x00", "")

    @staticmethod
    def _remove_unwanted_control_characters(text: str) -> str:
        """Remove control characters while preserving tabs and newlines."""
        return "".join(
            char
            for char in text
            if char in "\n\t"
            or char >= " "
            or char == "\x00"
        )

    @staticmethod
    def _clean_source_code(text: str) -> str:
        """
        Preserve indentation, internal whitespace and blank lines.
        Only remove trailing whitespace at the end of the file.
        """
        return text.rstrip()

    @staticmethod
    def _clean_html(text: str) -> str:
        """
        Preserve HTML markup and indentation.
        Do not strip tags or collapse whitespace between elements.
        """
        return text.rstrip()
