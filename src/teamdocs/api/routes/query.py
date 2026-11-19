from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from teamdocs.api.deps import get_resources
from teamdocs.api.schemas import ErrorResponse, QueryRequest, QueryResponse, SourceOut
from teamdocs.services.container import AppResources

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about the indexed documents",
    responses={
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
def query(payload: QueryRequest, resources: AppResources = Depends(get_resources)) -> QueryResponse:
    """Retrieve the most relevant passages and (unless ``generate=false``) answer from them.

    Declared as a plain ``def`` so FastAPI runs the blocking embedding / vector-store / LLM
    calls in its worker thread pool instead of blocking the event loop.
    """
    result = resources.rag.answer(
        payload.question,
        top_k=payload.top_k,
        mode=payload.mode,
        metadata_filter={"filename": payload.filename} if payload.filename else None,
        generate=payload.generate,
    )
    return QueryResponse(
        answer=result.answer,
        sources=[SourceOut(**asdict(s)) for s in result.sources],
        insufficient_evidence=result.insufficient_evidence,
        generated=result.generated,
        mode=result.mode,
        model=result.model,
        timings_ms=result.timings_ms,
    )
