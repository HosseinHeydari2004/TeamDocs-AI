import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

from .base import BaseCleaner


class DocumentCleaner(BaseCleaner):
    """Clean extracted text documents while preserving metadata."""

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".doc",
        ".docx",
        ".docs",
        ".md",
        ".markdown",
        ".txt",
    }

    def clean(self, documents: list[Document]) -> list[Document]:
        cleaned_documents = []

        for document in documents:
            text = document.page_content

            if not isinstance(text, str) or not text.strip():
                continue

            metadata = document.metadata.copy()
            file_type = self._get_file_type(metadata)

            text = self._normalize_unicode(text)
            text = self._remove_control_characters(text)
            text = self._normalize_line_endings(text)

            if file_type == ".markdown":
                text = self._clean_markdown(text)
            else:
                text = self._clean_plain_text(text)

            text = self._normalize_whitespace(text)
            text = text.strip()

            if not text:
                continue

            cleaned_documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return cleaned_documents

    @staticmethod
    def _get_file_type(metadata: dict) -> str:
        """Infer the extension from metadata."""

        file_type = metadata.get("file_type")

        if file_type:
            extension = str(file_type).lower().strip()
            if not extension.startswith("."):
                extension = f".{extension}"

            if extension in DocumentCleaner.SUPPORTED_EXTENSIONS:
                return extension

        source = metadata.get("source", "")

        if source:
            extension = Path(str(source)).suffix.lower()

            if extension in DocumentCleaner.SUPPORTED_EXTENSIONS:
                return extension

        return ".txt"

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize equivalent Unicode representations."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """Remove unwanted control characters but preserve tabs/newlines."""
        return "".join(
            char
            for char in text
            if unicodedata.category(char) != "Cc"
            or char in "\n\r\t"
        )

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        """Standardize Windows, Unix and legacy line endings."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _clean_plain_text(text: str) -> str:
        """Clean extracted text without merging separate paragraphs."""

        # Remove null bytes and common zero-width formatting artifacts.
        text = text.replace("\x00", "")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")

        # Remove trailing whitespace on each line.
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Collapse excessive blank lines.
        text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

        # Normalize horizontal whitespace without merging words.
        text = re.sub(r"[ \t]+", " ", text)

        return text

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Clean Markdown while preserving its structural whitespace."""

        text = text.replace("\x00", "")
        text = text.replace("\ufeff", "")

        # Remove trailing spaces, but preserve indentation.
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        # Keep intentional Markdown structure.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Final whitespace normalization."""
        text = re.sub(r"\n[ \t]+\n", "\n\n", text)
        return text.strip()
