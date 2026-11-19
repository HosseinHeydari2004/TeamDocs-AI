from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from teamdocs import __version__
from teamdocs.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """The process is up. Says nothing about model/vector-store readiness (see /ready)."""
    return HealthResponse(version=__version__)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    responses={503: {"model": ReadyResponse}},
)
def ready(request: Request):
    resources = getattr(request.app.state, "resources", None)
    startup_error = getattr(request.app.state, "startup_error", None)
    if resources is None:
        body = ReadyResponse(
            ready=False,
            checks={
                "embedding_model": False,
                "vector_store": False,
                "bm25_index": False,
                "llm_configured": False,
                "error": startup_error or "Resources are still initialising.",
            },
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    try:
        chunks = resources.vectordb.count()
        store_ok = True
    except Exception:
        chunks, store_ok = None, False
    body = ReadyResponse(
        ready=store_ok,
        checks={
            "embedding_model": True,
            "vector_store": store_ok,
            "bm25_index": True,
            "llm_configured": resources.models.configured,
        },
        documents=len(resources.ingestion.list_documents()) if store_ok else None,
        chunks=chunks,
        startup_ms=resources.startup_ms,
    )
    return JSONResponse(status_code=200 if store_ok else 503, content=body.model_dump())
