from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from teamdocs.api.deps import get_resources
from teamdocs.api.schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentOut,
    ErrorResponse,
    IngestResponse,
    UrlIngestRequest,
)
from teamdocs.core.config import ALLOWED_UPLOAD_EXTENSIONS
from teamdocs.services.container import AppResources
from teamdocs.services.errors import FileTooLargeError, InvalidFileError, UnsupportedFileTypeError
from teamdocs.services.ingestion import IngestResult, safe_filename

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

_ERRORS = {413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}
# Cheap content sniffing so that e.g. an .exe renamed to .pdf is rejected before parsing.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
}


def _ingest_response(result: IngestResult) -> IngestResponse:
    return IngestResponse(
        status=result.status,
        doc_id=result.doc_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
        replaced_chunks=result.replaced_chunks,
        message=result.message,
        duration_ms=result.duration_ms,
    )


@router.post("", response_model=IngestResponse, summary="Upload and index a document", responses=_ERRORS)
async def upload_document(
    file: UploadFile = File(..., description="PDF, DOCX, PPTX, Markdown, text or source-code file."),
    resources: AppResources = Depends(get_resources),
) -> IngestResponse:
    """Validate, store and index one file. Re-uploading identical content is a no-op;
    uploading changed content under the same filename replaces the old version."""
    settings = resources.settings
    filename = safe_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}."
        )

    limit = settings.max_upload_bytes
    written = 0
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension, dir=settings.tmp_dir) as tmp:
        tmp_path = Path(tmp.name)
        try:
            while block := await file.read(1024 * 1024):
                written += len(block)
                if written > limit:
                    raise FileTooLargeError(f"The file exceeds the {settings.max_upload_mb} MB upload limit.")
                tmp.write(block)
        except Exception:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        if written == 0:
            raise InvalidFileError("The uploaded file is empty.")
        signatures = _SIGNATURES.get(extension)
        if signatures:
            with tmp_path.open("rb") as handle:
                head = handle.read(8)
            if not head.startswith(signatures):
                raise InvalidFileError(f"The file content does not look like a valid {extension} file.")
        result = await run_in_threadpool(resources.ingestion.ingest_file, tmp_path, filename)
    finally:
        tmp_path.unlink(missing_ok=True)
    return _ingest_response(result)


@router.post(
    "/url",
    response_model=IngestResponse,
    summary="Download a web page and index it",
    responses={**_ERRORS, 502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def ingest_url(payload: UrlIngestRequest, resources: AppResources = Depends(get_resources)) -> IngestResponse:
    """Fetch the page with the existing Web Loader (SSRF-protected: private/internal addresses are refused),
    clean, chunk, embed and store it. The URL is kept as the source of every passage."""
    result = await run_in_threadpool(resources.ingestion.ingest_url, payload.url)
    return _ingest_response(result)


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
def list_documents(resources: AppResources = Depends(get_resources)) -> DocumentListResponse:
    documents = resources.ingestion.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentOut(
                doc_id=d.doc_id,
                filename=d.filename,
                file_type=d.file_type,
                chunk_count=d.chunk_count,
                size_bytes=d.size_bytes,
                ingested_at=d.ingested_at,
                pages=d.pages,
            )
            for d in documents
        ],
        total_documents=len(documents),
        total_chunks=sum(d.chunk_count for d in documents),
    )


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Remove a document from the index",
    responses={404: {"model": ErrorResponse}},
)
def delete_document(doc_id: str, resources: AppResources = Depends(get_resources)) -> DeleteResponse:
    deleted = resources.ingestion.delete_document(doc_id)
    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)


@router.post(
    "/{doc_id}/reindex",
    response_model=IngestResponse,
    summary="Re-run the pipeline on the stored original",
    responses={404: {"model": ErrorResponse}},
)
def reindex_document(doc_id: str, resources: AppResources = Depends(get_resources)) -> IngestResponse:
    return _ingest_response(resources.ingestion.reindex_document(doc_id))
