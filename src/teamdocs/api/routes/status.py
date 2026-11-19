from __future__ import annotations

import shutil

import httpx
from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from teamdocs.api.deps import get_resources
from teamdocs.api.schemas import ErrorResponse, LLMSelection, StatusResponse
from teamdocs.core.config import reload_settings
from teamdocs.services.container import AppResources
from teamdocs.services.errors import InvalidQueryError

router = APIRouter(prefix="/api/v1", tags=["status"])


def _check_github(resources: AppResources) -> dict:
    """Live check of the token against the GitHub REST API (independent of OpenRouter)."""
    s = resources.models.settings
    token = s.github_personal_access_token
    if token is None:
        return {"state": "not_configured", "detail": "GITHUB_PERSONAL_ACCESS_TOKEN is not set."}
    try:
        response = httpx.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token.get_secret_value()}", "Accept": "application/vnd.github+json"},
            timeout=8.0,
        )
    except httpx.TimeoutException:
        return {"state": "unreachable", "detail": "GitHub did not answer in time."}
    except httpx.HTTPError as exc:
        return {"state": "unreachable", "detail": f"Cannot reach GitHub ({type(exc).__name__})."}
    if response.status_code == 200:
        return {"state": "ok", "detail": "Token accepted by GitHub."}
    if response.status_code in (401, 403):
        return {"state": "invalid_token", "detail": "GitHub rejected the configured token."}
    return {"state": "error", "detail": f"GitHub answered HTTP {response.status_code}."}


def _github_info(resources: AppResources, check: bool) -> dict:
    s = resources.models.settings
    uses_docker = not s.github_mcp_command
    info = {
        "configured": s.github_configured,
        "key_source": s.key_source("github_personal_access_token"),
        "mcp_runtime": "docker" if uses_docker else "local binary",
        "mcp_runtime_found": shutil.which("docker" if uses_docker else s.github_mcp_command) is not None,
        "state": "not_configured" if not s.github_configured else "not_checked",
        "detail": "",
    }
    if check:
        info.update(_check_github(resources))
    return info


def _status(resources: AppResources, check: bool) -> StatusResponse:
    try:
        chunks = resources.vectordb.count()
        docs = len(resources.ingestion.list_documents())
        ready = True
    except Exception:
        chunks = docs = None
        ready = False
    return StatusResponse(
        ready=ready, checked=check, llm=resources.models.status(check=check),
        github=_github_info(resources, check), documents=docs, chunks=chunks,
    )


@router.get("/status", response_model=StatusResponse, summary="Real status of OpenRouter, Ollama and GitHub",
            description="`check=true` performs live connectivity/credential probes (slower). No secret is ever returned.")
async def status(check: bool = False, resources: AppResources = Depends(get_resources)) -> StatusResponse:
    return await run_in_threadpool(_status, resources, check)


@router.put("/settings/llm", response_model=StatusResponse, summary="Select provider / models at runtime",
            responses={422: {"model": ErrorResponse}})
async def select_llm(payload: LLMSelection, resources: AppResources = Depends(get_resources)) -> StatusResponse:
    """Kept in memory only (not written to .env). Send 'default' for a model to clear its override."""
    try:
        resources.models.set_override(
            provider=payload.provider, openrouter_model=payload.openrouter_model, ollama_model=payload.ollama_model
        )
    except ValueError as exc:
        raise InvalidQueryError(str(exc)) from exc
    return await run_in_threadpool(_status, resources, False)


@router.post("/settings/reload", response_model=StatusResponse, summary="Re-read .env / environment",
             description="Applies a changed OPENROUTER_API_KEY / GITHUB_PERSONAL_ACCESS_TOKEN / Ollama settings without a restart.")
async def reload(resources: AppResources = Depends(get_resources)) -> StatusResponse:
    resources.models.reload(reload_settings())
    return await run_in_threadpool(_status, resources, False)
