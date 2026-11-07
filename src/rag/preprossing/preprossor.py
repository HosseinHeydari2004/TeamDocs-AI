from __future__ import annotations
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import Iterable
from langchain_openai import ChatOpenAI

from langchain_core.documents import Document

from src.rag.loaders.pipeline import LoaderPipeline
from src.rag.cleaners.pipelline import CleanerPipeline
from src.rag.chunker.pipeline import ChunkerPipeline
from src.rag.metadata_extraction.extractor import MetadataExtractor
load_dotenv()
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end RAG preprocessing: load -> clean -> metadata -> chunk."""

    def __init__(
            self,
            *,
            loader: LoaderPipeline | None = None,
            cleaner: CleanerPipeline | None = None,
            metadata_extractor: MetadataExtractor | None = None,
            chunker: ChunkerPipeline | None = None,
            chunk_size: int = 1200,
            overlap: int = 200,
            continue_on_metadata_error: bool = False,
    ) -> None:
        self.loader = loader or LoaderPipeline()
        self.cleaner = cleaner or CleanerPipeline()
        self._model = ChatOpenAI(
            model="deepseek-flash",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        self.metadata_extractor = (
            metadata_extractor
            if metadata_extractor is not None
            else MetadataExtractor(llm=self._model)
        )
        self.chunker = chunker or ChunkerPipeline(
            chunk_size=chunk_size,
            overlap=overlap,
        )
        self.continue_on_metadata_error = continue_on_metadata_error

    def run(self, source: str | Path | Iterable[str | Path]) -> list[Document]:
        """Run all preprocessing stages and return chunked Documents."""
        documents = self.load(source=source)
        # return documents
        documents = self.clean(documents)

        if not documents:
            logger.warning("No documents remain after loading and cleaning.")
            return []

        documents = self.extract_metadata(documents=documents)
        chunks = self.chunk(documents=documents)

        logger.info(
            "Preprocessing completed: documents=%d, chunks=%d",
            len(documents),
            len(chunks),
        )
        return chunks

    def load(self, source: str | Path | Iterable[str | Path]) -> list[Document]:
        documents = self.loader.load(source)
        logger.info("Loaded %d document units", len(documents))
        return documents

    def clean(self, documents: list[Document]) -> list[Document]:
        cleaned = self.cleaner.clean(documents=documents)
        logger.info("Cleaned documents: %d -> %d", len(documents), len(cleaned))
        return cleaned

    def extract_metadata(self, documents: list[Document]) -> list[Document]:
        # Metadata extraction is optional so deterministic preprocessing can
        # still run when no LLM has been configured.
        if self.metadata_extractor is None:
            logger.info("Metadata extractor not configured; skipping semantic metadata.")
            return documents

        enriched = self.metadata_extractor.extract(
            documents,
            continue_on_error=self.continue_on_metadata_error,
        )
        logger.info("Semantic metadata extracted for %d document units", len(enriched))
        return enriched

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self.chunker.chunk(documents)
        logger.info("Chunked %d document units into %d chunks", len(documents), len(chunks))
        return chunks
