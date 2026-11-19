from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document

from rag.chunker.pipeline import ChunkerPipeline
from rag.cleaners.pipelline import CleanerPipeline
from rag.loaders.pipeline import LoaderPipeline
from rag.metadata_extraction.extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end RAG preprocessing: load -> clean -> chunk -> (optional) metadata.

    Semantic metadata extraction is optional and needs an LLM. It runs **once per
    source file after chunking** and is copied to every chunk of that source,
    instead of once per loaded element (which meant one LLM call per paragraph).
    """

    def __init__(
            self,
            *,
            loader: LoaderPipeline | None = None,
            model=None,
            cleaner: CleanerPipeline | None = None,
            metadata_extractor: MetadataExtractor | None = None,
            chunker: ChunkerPipeline | None = None,
            chunk_size: int = 1200,
            overlap: int = 200,
            continue_on_metadata_error: bool = True,
    ) -> None:
        self.loader = loader or LoaderPipeline()
        self.cleaner = cleaner or CleanerPipeline()
        self._model = model

        if metadata_extractor is not None:
            self.metadata_extractor: MetadataExtractor | None = metadata_extractor
        elif model is not None:
            self.metadata_extractor = MetadataExtractor(llm=model)
        else:
            self.metadata_extractor = None

        self.chunker = chunker or ChunkerPipeline(chunk_size=chunk_size, overlap=overlap)
        self.continue_on_metadata_error = continue_on_metadata_error

    def run(self, source: str | Path | Iterable[str | Path]) -> list[Document]:
        """Run all preprocessing stages and return chunked Documents."""
        documents = self.load(source=source)
        return self.process_documents(documents)

    def process_documents(self, documents: list[Document]) -> list[Document]:
        """Clean, chunk and enrich already-loaded documents."""
        documents = self.clean(documents)

        if not documents:
            logger.warning("No documents remain after loading and cleaning.")
            return []

        chunks = self.chunk(documents=documents)
        chunks = self.extract_metadata(documents=chunks)

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
        if self.metadata_extractor is None:
            logger.info("Metadata extractor not configured; skipping semantic metadata.")
            return documents

        enriched = self.metadata_extractor.extract_by_source(
            documents,
            continue_on_error=self.continue_on_metadata_error,
        )
        logger.info("Semantic metadata attached to %d chunks", len(enriched))
        return enriched

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self.chunker.chunk(documents)
        logger.info("Chunked %d document units into %d chunks", len(documents), len(chunks))
        return chunks
