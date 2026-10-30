import re
import unicodedata

from langchain_core.documents import Document
from .base import BaseCleaner


class WebDocumentCleaner(BaseCleaner):
    """Clean English-oriented documents extracted from web pages."""

    def clean(self, documents: list[Document]) -> list[Document]:
        cleaned_documents = []

        for document in documents:
            text = document.page_content

            if not isinstance(text, str) or not text.strip():
                continue

            text = self._normalize_unicode(text)
            text = self._normalize_line_endings(text)
            text = self._remove_control_characters(text)
            text = self._remove_invisible_characters(text)
            text = self._normalize_whitespace(text)
            text = text.strip()

            if not text:
                continue

            metadata = document.metadata.copy()
            metadata["content_type"] = "web"
            metadata.setdefault("source_type", "web")

            cleaned_documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return cleaned_documents

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        return "".join(
            char
            for char in text
            if unicodedata.category(char) != "Cc"
            or char in "\n\t"
        )

    @staticmethod
    def _remove_invisible_characters(text: str) -> str:
        return text.translate({
            ord("\u200b"): None,
            ord("\ufeff"): None,
            ord("\u00ad"): None,
        })

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

        return text
