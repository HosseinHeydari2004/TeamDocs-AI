"""Retrieval + grounded answer generation with structured, verifiable citations."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from devmind.agents.errors import describe_failure
from devmind.llm import ModelManager, ModelUnavailableError
from rag.vector_db.vector_database import InvalidFilterError, RetrievedChunk, VectorDatabase
from teamdocs.core.config import Settings
from teamdocs.services.errors import (
    InvalidQueryError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    RetrievalError,
)
from teamdocs.services.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    NOT_FOUND_TOKEN,
)

logger = logging.getLogger(__name__)

Mode = Literal["hybrid", "dense", "bm25"]
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The indexed documents do not contain enough information to answer this question."
)
_CITATION = re.compile(r"\[(\d{1,3})\]")
_EXCERPT_LIMIT = 1500


@dataclass(slots=True)
class Source:
    index: int  # the [n] used in the answer
    chunk_id: str
    doc_id: str | None
    filename: str
    page: int | None
    section: str | None
    excerpt: str
    score: float
    dense_score: float | None
    bm25_score: float | None
    cited: bool = False
    source_url: str | None = None  # set for web pages; lets the UI link to the real origin


@dataclass(slots=True)
class AnswerResult:
    answer: str | None
    sources: list[Source]
    mode: str
    generated: bool
    insufficient_evidence: bool
    model: str | None
    timings_ms: dict[str, float] = field(default_factory=dict)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content)


def translate_llm_error(exc: Exception) -> Exception:
    """Map provider/SDK exceptions to user-safe domain errors (never leaks keys or payloads)."""
    failure = describe_failure(exc)
    cls = {
        "llm_timeout": LLMTimeoutError,
        "llm_rate_limited": LLMRateLimitError,
        "llm_not_configured": LLMNotConfiguredError,
    }.get(failure.code, LLMProviderError)
    error = cls(failure.message)
    error.code = failure.code  # precise code (llm_auth, llm_unreachable, ...) for the client
    return error


class RagService:
    def __init__(
        self,
        vectordb: VectorDatabase,
        chat_model: Any | None,
        settings: Settings,
        models: ModelManager | None = None,
    ) -> None:
        self._db = vectordb
        self._llm = chat_model
        self._settings = settings
        self._models = models  # when set, provider/model are chosen per request (OpenRouter -> Ollama policy)

    # --------------------------------------------------------------- retrieval
    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: Mode | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self._settings
        k = top_k or cfg.top_k
        mode = mode or cfg.retrieval_mode
        try:
            if mode == "dense":
                chunks = self._db.similarity_search_with_scores(question, k, metadata_filter)
            elif mode == "bm25":
                chunks = self._db.bm25_search_with_scores(question, k, metadata_filter)
            else:
                chunks = self._db.hybrid_search_with_scores(
                    question,
                    k,
                    fetch_k=max(cfg.fetch_k, k),
                    metadata_filter=metadata_filter,
                    dense_weight=cfg.dense_weight,
                )
        except InvalidFilterError as exc:
            raise InvalidQueryError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise RetrievalError("Retrieval failed. Check the server logs.") from exc
        return self._apply_evidence_threshold(chunks)

    def _apply_evidence_threshold(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop chunks with neither lexical support nor sufficient dense similarity."""
        threshold = self._settings.min_dense_similarity
        if threshold is None:
            return chunks
        return [
            c
            for c in chunks
            if c.bm25_score is not None or (c.dense_score is not None and c.dense_score >= threshold)
        ]

    # ------------------------------------------------------------------ answer
    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: Mode | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        generate: bool = True,
    ) -> AnswerResult:
        question = (question or "").strip()
        if not question:
            raise InvalidQueryError("The question must not be empty.")
        if generate and self._llm is None and self._models is None:
            raise LLMNotConfiguredError(
                "No language model is configured (set OPENROUTER_API_KEY), "
                "or send generate=false to retrieve sources only."
            )

        mode = mode or self._settings.retrieval_mode
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        chunks = self.retrieve(question, top_k=top_k, mode=mode, metadata_filter=metadata_filter)
        timings["retrieval"] = _ms(t0)
        sources = [self._to_source(i, c) for i, c in enumerate(chunks, start=1)]

        if not generate:
            return AnswerResult(None, sources, mode, False, not sources, None, timings)

        if not sources:  # nothing retrieved -> do not spend an LLM call, do not guess
            return AnswerResult(INSUFFICIENT_EVIDENCE_MESSAGE, [], mode, True, True, self._model_name(), timings)

        t1 = time.perf_counter()
        raw_answer, model_name = self._generate(question, chunks)
        timings["generation"] = _ms(t1)
        timings["total"] = _ms(t0)

        if raw_answer.strip().strip("`'\" .").upper() == NOT_FOUND_TOKEN or not raw_answer.strip():
            return AnswerResult(INSUFFICIENT_EVIDENCE_MESSAGE, sources, mode, True, True, model_name, timings)

        answer, cited = self._sanitize_citations(raw_answer, len(sources))
        for source in sources:
            source.cited = source.index in cited
        return AnswerResult(answer, sources, mode, True, False, model_name, timings)

    # ----------------------------------------------------------------- helpers
    def _generate(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, str | None]:
        context = self._build_context(chunks)
        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=ANSWER_USER_TEMPLATE.format(context=context, question=question)),
        ]
        try:
            if self._models is not None and self._llm is None:
                response, handle = self._models.run("rag", lambda h: h.model.invoke(messages))
                return _message_text(response.content), f"{handle.choice.provider}:{handle.choice.model}"
            response = self._llm.invoke(messages)  # blocking; endpoints run in a worker thread
        except ModelUnavailableError as exc:
            raise LLMNotConfiguredError(str(exc)) from exc
        except Exception as exc:
            logger.warning("LLM call failed: %s", type(exc).__name__)
            raise translate_llm_error(exc) from exc
        return _message_text(response.content), self._model_name()

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        budget = self._settings.max_context_chars
        parts: list[str] = []
        used = 0
        for index, chunk in enumerate(chunks, start=1):
            meta = chunk.document.metadata
            label = f"[{index}] source: {meta.get('filename', meta.get('source', 'unknown'))}"
            if meta.get("page_number") is not None:
                label += f", page {meta['page_number']}"
            if meta.get("section"):
                label += f", section: {meta['section']}"
            text = chunk.document.page_content.strip()
            remaining = budget - used - len(label) - 2
            if remaining <= 200 and parts:  # no room for a meaningful passage
                break
            if len(text) > remaining:
                text = text[: max(remaining, 200)] + " ..."
            parts.append(f"{label}\n{text}")
            used += len(parts[-1])
        return "\n\n".join(parts)

    @staticmethod
    def _sanitize_citations(answer: str, source_count: int) -> tuple[str, set[int]]:
        """Remove ``[n]`` markers that do not refer to a retrieved passage; return cited numbers."""
        cited: set[int] = set()

        def replace(match: re.Match[str]) -> str:
            number = int(match.group(1))
            if 1 <= number <= source_count:
                cited.add(number)
                return match.group(0)
            return ""  # fabricated citation: strip it

        return _CITATION.sub(replace, answer).strip(), cited

    @staticmethod
    def _to_source(index: int, chunk: RetrievedChunk) -> Source:
        meta = chunk.document.metadata
        page = meta.get("page_number")
        excerpt = chunk.document.page_content.strip()
        if len(excerpt) > _EXCERPT_LIMIT:
            excerpt = excerpt[:_EXCERPT_LIMIT].rstrip() + " ..."
        return Source(
            index=index,
            chunk_id=chunk.id,
            doc_id=meta.get("doc_id"),
            filename=str(meta.get("filename", meta.get("source", "unknown"))),
            page=int(page) if isinstance(page, (int, float)) else None,
            section=meta.get("section") or None,
            excerpt=excerpt,
            score=round(chunk.score, 6),
            dense_score=None if chunk.dense_score is None else round(chunk.dense_score, 4),
            bm25_score=None if chunk.bm25_score is None else round(chunk.bm25_score, 4),
            source_url=meta.get("source_url") or None,
        )

    # Public aliases: the RAG sub-agent reuses exactly the same citation logic.
    to_source = _to_source
    sanitize_citations = _sanitize_citations

    def _model_name(self) -> str | None:
        if self._llm is not None:
            return self._settings.openrouter_model
        if self._models is not None:
            try:
                return self._models.choose("rag").model
            except ModelUnavailableError:
                return None
        return None


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
