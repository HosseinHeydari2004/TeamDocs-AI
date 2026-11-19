from __future__ import annotations

from fastapi import APIRouter, Depends

from devmind.agents.main_agent import MainAgentError
from teamdocs.api.deps import get_resources
from teamdocs.api.schemas import (
    AgentErrorOut,
    AgentRunOut,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    GitHubRefOut,
    SourceOut,
)
from teamdocs.services.container import AppResources
from teamdocs.services.errors import (
    AgentFailedError,
    AgentTimeoutError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    TeamDocsError,
)

router = APIRouter(prefix="/api/v1", tags=["chat"])

_ERROR_CLASSES: dict[str, type[TeamDocsError]] = {
    "llm_not_configured": LLMNotConfiguredError,
    "llm_rate_limited": LLMRateLimitError,
    "llm_timeout": LLMTimeoutError,
    "agent_timeout": AgentTimeoutError,
    "agent_limit": AgentFailedError,
    "agent_error": AgentFailedError,
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Talk to the Main Agent",
    responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
               503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def chat(payload: ChatRequest, resources: AppResources = Depends(get_resources)) -> ChatResponse:
    """The single entry point for questions. The Main Agent (Deep Agents) decides whether to call the RAG
    sub-agent (documents + web pages), the GitHub sub-agent (repositories, issues, PRs), both, or to ask you a
    clarifying question. Sub-agent failures are reported in ``agents_used``/``warnings``, never hidden."""
    try:
        outcome = await resources.main_agent.chat(
            payload.message, [turn.model_dump() for turn in payload.history]
        )
    except MainAgentError as exc:
        error = _ERROR_CLASSES.get(exc.code, LLMProviderError)(exc.message)
        error.code = exc.code
        raise error from exc

    sources, seen = [], set()
    for result in outcome.results:
        for src in result.sources:
            if src.index not in seen:
                seen.add(src.index)
                sources.append(SourceOut(**src.model_dump(exclude={"source_url"}), source_url=src.source_url))
    sources.sort(key=lambda s: s.index)
    github = [GitHubRefOut(**g.model_dump()) for r in outcome.results for g in r.github]
    return ChatResponse(
        answer=outcome.answer,
        agents_used=[
            AgentRunOut(
                agent=r.agent, ok=r.ok, tool_calls=r.tool_calls, model=r.model, duration_ms=r.duration_ms,
                insufficient_evidence=r.insufficient_evidence,
                error=AgentErrorOut(**r.error.model_dump()) if r.error else None,
            )
            for r in outcome.results
        ],
        sources=sources,
        github=github,
        warnings=outcome.warnings,
        model=outcome.model,
        duration_ms=outcome.duration_ms,
    )
