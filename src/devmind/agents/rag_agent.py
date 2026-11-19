"""RAG sub-agent: agentic retrieval over the shared vector store, with structured citations."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool

from devmind.agents.common import final_answer, limits
from devmind.agents.contracts import SourceRef, SubAgentError, SubAgentResult
from devmind.agents.errors import describe_failure
from devmind.agents.prompts import RAG_SUBAGENT_PROMPT
from devmind.llm import ModelManager
from teamdocs.services.answering import INSUFFICIENT_EVIDENCE_MESSAGE, RagService, Source

logger = logging.getLogger(__name__)
_MODES = {"hybrid": "hybrid", "semantic": "dense", "keyword": "bm25"}
_PASSAGE_LIMIT = 1500


@dataclass
class SourceRegistry:
    """Passage numbers shared by all RAG calls of one chat turn, so ``[n]`` is never ambiguous."""

    sources: list[Source] = field(default_factory=list)
    _by_chunk: dict[str, int] = field(default_factory=dict)

    def add(self, source: Source) -> Source:
        known = self._by_chunk.get(source.chunk_id)
        if known is not None:
            return self.sources[known - 1]
        source.index = len(self.sources) + 1
        self.sources.append(source)
        self._by_chunk[source.chunk_id] = source.index
        return source


def build_rag_tools(rag: RagService, registry: SourceRegistry, touched: set[int], calls: list[str],
                    errors: list[str], document: str | None, max_chars: int) -> list[BaseTool]:
    @tool
    def search_knowledge_base(query: str, mode: Literal["hybrid", "semantic", "keyword"] = "hybrid") -> str:
        """Search the indexed documents and imported web pages. Returns numbered passages [n] with their source."""
        calls.append(f"search_knowledge_base[{mode}]")
        try:
            chunks = rag.retrieve(
                query, mode=_MODES.get(mode, "hybrid"),
                metadata_filter={"filename": document} if document else None,
            )
        except Exception as exc:  # TeamDocsError (bad filter, retrieval failure) or vector store error
            errors.append(type(exc).__name__)
            logger.warning("RAG retrieval failed: %s", type(exc).__name__)
            return "ERROR: retrieval failed."
        if not chunks:
            return "No relevant passages found."
        parts: list[str] = []
        used = 0
        for chunk in chunks:
            src = registry.add(rag.to_source(0, chunk))
            touched.add(src.index)
            label = f"[{src.index}] source: {src.source_url or src.filename}"
            if src.page:
                label += f", page {src.page}"
            if src.section:
                label += f", section: {src.section}"
            text = chunk.document.page_content.strip()[:_PASSAGE_LIMIT]
            if used + len(text) > max_chars and parts:
                break
            parts.append(f"{label}\n{text}")
            used += len(text)
        return "\n\n".join(parts)

    return [search_knowledge_base]


class RagSubAgent:
    name = "rag"

    def __init__(self, rag: RagService, models: ModelManager) -> None:
        self._rag = rag
        self._models = models

    async def run(self, question: str, document: str | None = None, registry: SourceRegistry | None = None) -> SubAgentResult:
        started = time.perf_counter()
        settings = self._models.settings
        registry = registry or SourceRegistry()
        touched: set[int] = set()
        calls: list[str] = []
        errors: list[str] = []
        tools = build_rag_tools(self._rag, registry, touched, calls, errors, document, settings.max_context_chars)

        async def attempt(handle):
            agent = create_agent(
                model=handle.model, tools=tools, system_prompt=RAG_SUBAGENT_PROMPT,
                middleware=limits(settings.sub_agent_max_tool_calls),
            )
            return await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=question)]},
                              config={"recursion_limit": settings.agent_recursion_limit}),
                settings.agent_timeout_seconds,
            )

        try:
            result, handle = await self._models.arun("rag", attempt)
        except Exception as exc:
            failure = describe_failure(exc)
            return SubAgentResult(agent="rag", ok=False, error=SubAgentError(code=failure.code, message=failure.message),
                                  tool_calls=calls, duration_ms=_ms(started))

        used = sorted(touched)
        if not used:
            # The model answered without evidence (or retrieval found nothing / failed): never pass that on as knowledge.
            if errors:
                return SubAgentResult(agent="rag", ok=False, model=handle.describe(), tool_calls=calls,
                                      error=SubAgentError(code="retrieval_failed", message="Retrieval failed. Check the server logs."),
                                      duration_ms=_ms(started))
            return SubAgentResult(agent="rag", ok=True, answer=INSUFFICIENT_EVIDENCE_MESSAGE, insufficient_evidence=True,
                                  model=handle.describe(), tool_calls=calls, duration_ms=_ms(started))

        answer, cited = self._rag.sanitize_citations(final_answer(result), len(registry.sources))
        if not answer:
            return SubAgentResult(agent="rag", ok=False, model=handle.describe(), tool_calls=calls,
                                  error=SubAgentError(code="agent_limit", message="The RAG agent did not produce an answer within its limits."),
                                  duration_ms=_ms(started))
        sources = []
        for index in used:
            src = registry.sources[index - 1]
            src.cited = index in cited
            sources.append(SourceRef(**asdict(src)))
        return SubAgentResult(agent="rag", ok=True, answer=answer, sources=sources, model=handle.describe(),
                              tool_calls=calls, duration_ms=_ms(started))


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
