"""Application resources, created once at startup and shared by all requests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings

from devmind.agents.github_agent import GitHubSubAgent
from devmind.agents.main_agent import MainAgentService
from devmind.agents.rag_agent import RagSubAgent
from devmind.llm import ModelManager
from rag.chunker.pipeline import ChunkerPipeline
from rag.loaders.pipeline import LoaderPipeline
from rag.preprossing.preprossor import PreprocessingPipeline
from rag.vector_db.vector_database import VectorDatabase, build_embeddings
from teamdocs.core.config import Settings
from teamdocs.services.answering import RagService
from teamdocs.services.ingestion import IngestionService

logger = logging.getLogger(__name__)


@dataclass
class AppResources:
    settings: Settings
    embeddings: Embeddings
    vectordb: VectorDatabase
    chat_model: Any | None  # injected model (tests/benchmarks); normally None -> ModelManager decides
    ingestion: IngestionService
    rag: RagService
    models: ModelManager
    main_agent: MainAgentService
    startup_ms: dict[str, float] = field(default_factory=dict)


def build_resources(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
    chat_model: Any | None = None,
    github_factory: Any | None = None,
) -> AppResources:
    """Create the embedding model, vector store, BM25 index and LLM client **once**.

    ``embeddings`` / ``chat_model`` can be injected (tests, benchmarks).
    """
    timings: dict[str, float] = {}
    started = time.perf_counter()
    settings.ensure_directories()

    t = time.perf_counter()
    if embeddings is None:
        embeddings = build_embeddings(
            settings.embedding_model, settings.model_cache_dir, settings.embedding_device
        )
    timings["embedding_model_init"] = _ms(t)

    t = time.perf_counter()
    vectordb = VectorDatabase(
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        bm25_weight=1.0 - settings.dense_weight,
        embeddings=embeddings,
    )
    timings["vector_store_init"] = _ms(t)

    t = time.perf_counter()
    indexed_chunks = vectordb.refresh_bm25(force=True)
    timings["bm25_index_build"] = _ms(t)

    t = time.perf_counter()
    vectordb.embed_query("warm-up")  # first inference is slower; pay it at startup, not on request #1
    timings["embedding_warmup"] = _ms(t)

    models = ModelManager(settings, fixed_model=chat_model)

    preprocessor = PreprocessingPipeline(
        loader=LoaderPipeline(
            per_suffix_kwargs={
                ".pdf": {"backend": settings.pdf_backend},
                "web": {
                    "timeout": settings.web_fetch_timeout_seconds,
                    "max_bytes": settings.web_max_bytes,
                    "max_redirects": settings.web_max_redirects,
                    "allow_private": settings.web_allow_private_hosts,
                },
            }
        ),
        chunker=ChunkerPipeline(settings.chunk_size, settings.chunk_overlap),
        model=models.get("rag").model if settings.metadata_extraction else None,
    )
    ingestion = IngestionService(vectordb, preprocessor, settings.upload_dir)
    rag = RagService(vectordb, chat_model, settings, models=models)
    main_agent = MainAgentService(models, RagSubAgent(rag, models), GitHubSubAgent(models, github_factory=github_factory))

    timings["total"] = _ms(started)
    logger.info(
        "Resources ready in %.0f ms (chunks indexed: %d, llm configured: %s)",
        timings["total"],
        indexed_chunks,
        models.configured,
    )
    return AppResources(settings, embeddings, vectordb, chat_model, ingestion, rag, models, main_agent, timings)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
