
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .base import BaseChunker


class CodeChunker(BaseChunker):
    LANGUAGE_MAP = {
        "python": Language.PYTHON,
        "py": Language.PYTHON,
        "javascript": Language.JS,
        "js": Language.JS,
        "typescript": Language.TS,
        "ts": Language.TS,
        "java": Language.JAVA,
        "cpp": Language.CPP,
        "c++": Language.CPP,
        "c": Language.C,
        "go": Language.GO,
        "rust": Language.RUST,
        "rustlang": Language.RUST,
        "php": Language.PHP,
        "ruby": Language.RUBY,
        "swift": Language.SWIFT,
        "kotlin": Language.KOTLIN,
        "scala": Language.SCALA,
        "html": Language.HTML,
    }

    EXTENSION_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sql": "sql",
        ".ddl": "sql",
        ".psql": "sql",
        ".pgsql": "sql",
        ".mysql": "sql",
    }

    SQL_SEPARATORS = [
        "\nCREATE ",
        "\nALTER ",
        "\nINSERT ",
        "\nUPDATE ",
        "\nDELETE ",
        "\nSELECT ",
        "\nWITH ",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size"
            )

        self._chunk_size = chunk_size
        self._overlap = overlap

    def _get_language_name(self, doc: Document) -> str | None:
        metadata = doc.metadata or {}

        language = metadata.get("language")
        if language:
            normalized = str(language).strip().lower().lstrip(".")
            if normalized in self.LANGUAGE_MAP or normalized == "sql":
                return normalized

        source = str(metadata.get("source", ""))
        extension = Path(source).suffix.lower()

        return self.EXTENSION_MAP.get(extension)

    def chunk(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue

            language_name = self._get_language_name(doc)

            if language_name == "sql":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._overlap,
                    separators=self.SQL_SEPARATORS,
                )

            elif language_name in self.LANGUAGE_MAP:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=self.LANGUAGE_MAP[language_name],
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._overlap,
                )

            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._overlap,
                    separators=[
                        "\nclass ",
                        "\ndef ",
                        "\nasync def ",
                        "\nfunction ",
                        "\n\n",
                        "\n",
                        " ",
                        "",
                    ],
                )

            code_document = Document(
                page_content=text,
                metadata={
                    **doc.metadata,
                    "element_type": "code",
                },
            )

            chunks.extend(splitter.split_documents([code_document]))

        return chunks
