"""Document ingestion: load -> clean -> chunk -> embed -> store, with update/delete.

Identity model
--------------
* ``doc_id``  = first 16 hex chars of the SHA-256 of the file *content*.
* chunk id    = ``<doc_id>:<chunk_index>`` (stable, so re-indexing is an upsert).
* A document is "the same" when its content hash matches -> it is **not** re-embedded.
* A new upload with the same *filename* but different content replaces the old
  version (old chunks are deleted after the new ones were stored successfully).
* The indexed documents are listed straight from chunk metadata, so there is no
  separate registry that could drift out of sync with the vector store.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document

from rag.loaders.pipeline import LoaderPipeline
from rag.loaders.web_loader.safe_http import WebFetchError
from rag.preprossing.preprossor import PreprocessingPipeline
from rag.vector_db.vector_database import VectorDatabase
from teamdocs.services.errors import (
    DocumentNotFoundError,
    EmptyDocumentError,
    IngestionError,
    InvalidFileError,
    InvalidUrlError,
    UrlFetchError,
    UrlNotAllowedError,
    UrlTimeoutError,
)

logger = logging.getLogger(__name__)

# Loader metadata that is safe and useful to keep (everything else - absolute
# paths, coordinates, ids - is dropped).
_KEEP_LOADER_METADATA = ("page", "page_number", "category")
_STATUS = Literal["indexed", "replaced", "unchanged", "duplicate"]
_STORED_NAME = re.compile(r"^[0-9a-f]{16}$")


@dataclass(slots=True)
class IngestResult:
    status: _STATUS
    doc_id: str
    filename: str
    chunk_count: int
    message: str
    replaced_chunks: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class DocumentInfo:
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    size_bytes: int | None
    ingested_at: str | None
    pages: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(name: str | None) -> str:
    """Reduce an client-supplied name to a safe basename (no paths, no control chars)."""
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip().strip(".")
    base = re.sub(r"\s+", " ", base)
    if not base:
        raise InvalidFileError("The file has no usable name.")
    return base[:200]


class IngestionService:
    def __init__(
        self,
        vectordb: VectorDatabase,
        preprocessor: PreprocessingPipeline,
        upload_dir: Path,
        loader: LoaderPipeline | None = None,
    ) -> None:
        self._db = vectordb
        self._preprocessor = preprocessor
        self._loader = loader or preprocessor.loader
        self._upload_dir = upload_dir
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        # One writer at a time: prevents two concurrent uploads of the same file
        # from both embedding it, and keeps delete+add sequences consistent.
        self._write_lock = threading.Lock()
        self._mutations = 0  # bumped on every write; keys the document-list cache
        self._list_cache: tuple[tuple[int, int], list[DocumentInfo]] | None = None

    # ------------------------------------------------------------------ public
    def ingest_file(self, path: Path, filename: str, *, force: bool = False) -> IngestResult:
        """Index ``path`` under the display name ``filename``."""
        started = time.perf_counter()
        filename = safe_filename(filename)
        extension = Path(filename).suffix.lower()

        with self._write_lock:
            file_hash = file_sha256(path)
            doc_id = file_hash[:16]

            if not force:
                same_content = self._db.get_metadatas({"doc_id": doc_id})
                if same_content:
                    existing_name = same_content[0][1].get("filename", filename)
                    status: _STATUS = "unchanged" if existing_name == filename else "duplicate"
                    message = (
                        "This exact file is already indexed; nothing was re-embedded."
                        if status == "unchanged"
                        else f"Identical content is already indexed as '{existing_name}'; nothing was re-embedded."
                    )
                    return IngestResult(
                        status,
                        doc_id,
                        str(existing_name),
                        len(same_content),
                        message,
                        duration_ms=_ms(started),
                    )

            chunks = self._build_chunks(path, filename, doc_id, file_hash, extension)
            return self._commit(chunks, filename, doc_id, started, force, lambda: self._store_original(path, doc_id, extension))

    def ingest_url(self, url: str, *, force: bool = False) -> IngestResult:
        """Download a web page with the existing WebLoader and index it like any other document.

        The page is identified by the hash of its extracted text: unchanged content is not re-embedded,
        changed content replaces the previous version of the same URL. The URL is kept in the metadata
        (``source_url``) so answers can show the real origin.
        """
        started = time.perf_counter()
        try:
            raw_docs = self._loader.load_url((url or "").strip())
        except WebFetchError as exc:
            if exc.code == "invalid_url":
                raise InvalidUrlError(exc.message) from exc
            if exc.code == "blocked_host":
                raise UrlNotAllowedError(exc.message) from exc
            if exc.code == "timeout":
                raise UrlTimeoutError(exc.message) from exc
            raise UrlFetchError(exc.message) from exc
        raw_docs = [d for d in raw_docs if d.page_content and d.page_content.strip()]
        if not raw_docs:
            raise EmptyDocumentError("No readable text was found at this URL (is it a JavaScript-only page?).")
        final_url = str(raw_docs[0].metadata.get("source_url") or url).strip()
        text = "\n\n".join(d.page_content for d in raw_docs)
        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = file_hash[:16]

        with self._write_lock:
            if not force:
                same = self._db.get_metadatas({"doc_id": doc_id})
                if same:
                    existing = str(same[0][1].get("filename", final_url))
                    status: _STATUS = "unchanged" if existing == final_url else "duplicate"
                    message = (
                        "This page is already indexed and its content has not changed; nothing was re-embedded."
                        if status == "unchanged"
                        else f"Identical content is already indexed as '{existing}'; nothing was re-embedded."
                    )
                    return IngestResult(status, doc_id, existing, len(same), message, duration_ms=_ms(started))
            prepared = [
                Document(
                    page_content=d.page_content,
                    metadata={
                        "source": final_url, "source_url": final_url, "filename": final_url, "doc_id": doc_id,
                        "file_hash": file_hash, "file_type": "web", "size_bytes": len(text.encode("utf-8")),
                        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    },
                )
                for d in raw_docs
            ]
            chunks = self._finalize_chunks(prepared, final_url)
            return self._commit(chunks, final_url, doc_id, started, force, lambda: None)

    def _commit(self, chunks: list[Document], filename: str, doc_id: str, started: float, force: bool, store_original) -> IngestResult:
        """Store ``chunks`` and drop the previous version of ``filename`` (caller holds the write lock)."""
        chunk_ids = [f"{doc_id}:{i:05d}" for i in range(len(chunks))]
        # Chunks to drop once the new ones are stored: previous versions of a document with this
        # display name, plus any stale tail left by a re-index that produced fewer chunks than before.
        new_ids = set(chunk_ids)
        old_ids = [cid for cid, _meta in self._db.get_metadatas({"filename": filename}) if cid not in new_ids]

        try:
            self._db.add_documents(chunks, ids=chunk_ids)
        except Exception as exc:  # embedding model / vector store failure
            logger.exception("Indexing failed for %s", filename)
            raise IngestionError("The document could not be embedded and stored. Check the server logs.") from exc

        if old_ids:
            self._db.delete_ids(old_ids)
            self._remove_orphaned_originals(keep_doc_id=doc_id)

        store_original()
        self._db.refresh_bm25()  # pay the BM25 rebuild now, not on the next query
        self._mutations += 1

        status = "replaced" if old_ids and not force else "indexed"
        logger.info("Ingested %s doc_id=%s chunks=%d status=%s", filename, doc_id, len(chunks), status)
        return IngestResult(
            status, doc_id, filename, len(chunks),
            "Document indexed." if status == "indexed" else "Document updated; the previous version was removed.",
            replaced_chunks=len(old_ids), duration_ms=_ms(started),
        )

    def list_documents(self) -> list[DocumentInfo]:
        # /ready (polled by the UI after every action) calls this: scanning all chunk metadata is O(chunks),
        # so the result is cached until a write happens or the chunk count changes behind our back.
        key = (self._mutations, self._db.count())
        cached = self._list_cache
        if cached is not None and cached[0] == key:
            return list(cached[1])
        documents = self._scan_documents()
        self._list_cache = (key, documents)
        return list(documents)

    def _scan_documents(self) -> list[DocumentInfo]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _cid, meta in self._db.get_metadatas():
            doc_id = meta.get("doc_id")
            if doc_id:
                grouped[str(doc_id)].append(meta)

        documents: list[DocumentInfo] = []
        for doc_id, metas in grouped.items():
            first = metas[0]
            pages = {m.get("page_number") for m in metas if isinstance(m.get("page_number"), int)}
            documents.append(
                DocumentInfo(
                    doc_id=doc_id,
                    filename=str(first.get("filename", first.get("source", doc_id))),
                    file_type=str(first.get("file_type", "")),
                    chunk_count=len(metas),
                    size_bytes=first.get("size_bytes"),
                    ingested_at=first.get("ingested_at"),
                    pages=len(pages) or None,
                )
            )
        documents.sort(key=lambda d: (d.ingested_at or "", d.filename), reverse=True)
        return documents

    def get_document(self, doc_id: str) -> DocumentInfo:
        for document in self.list_documents():
            if document.doc_id == doc_id:
                return document
        raise DocumentNotFoundError("Document not found.")

    def delete_document(self, doc_id: str) -> int:
        with self._write_lock:
            entries = self._db.get_metadatas({"doc_id": doc_id})
            if not entries:
                raise DocumentNotFoundError("Document not found.")
            deleted = self._db.delete_ids([cid for cid, _ in entries])
            for stored in self._upload_dir.glob(f"{doc_id}.*"):
                stored.unlink(missing_ok=True)
            self._db.refresh_bm25()
            self._mutations += 1
            logger.info("Deleted document doc_id=%s chunks=%d", doc_id, deleted)
            return deleted

    def reindex_document(self, doc_id: str) -> IngestResult:
        """Re-run the full pipeline on the stored original (e.g. after changing chunk settings)."""
        info = self.get_document(doc_id)
        if info.file_type == "web":
            return self.ingest_url(info.filename, force=True)  # web pages are re-downloaded
        stored = next(iter(self._upload_dir.glob(f"{doc_id}.*")), None)
        if stored is None:
            raise DocumentNotFoundError("The original file is no longer stored; upload it again.")
        return self.ingest_file(stored, info.filename, force=True)

    # ----------------------------------------------------------------- helpers
    def _build_chunks(
        self, path: Path, filename: str, doc_id: str, file_hash: str, extension: str
    ) -> list[Document]:
        try:
            raw_docs = self._loader.load_file(path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", filename, exc)
            raise InvalidFileError(
                "The file could not be read. It may be corrupted, encrypted or not a valid document."
            ) from exc

        raw_docs = [d for d in raw_docs if d.page_content and d.page_content.strip()]
        if not raw_docs:
            raise EmptyDocumentError(
                "No extractable text was found in the file (is it a scanned image or empty?)."
            )

        ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
        size_bytes = path.stat().st_size
        prepared: list[Document] = []
        for doc in raw_docs:
            meta = {k: doc.metadata[k] for k in _KEEP_LOADER_METADATA if k in doc.metadata}
            if "page_number" not in meta and isinstance(meta.get("page"), int):
                meta["page_number"] = meta["page"] + 1  # PyPDF pages are 0-based
            meta.update(
                source=filename,
                filename=filename,
                doc_id=doc_id,
                file_hash=file_hash,
                file_type=extension.lstrip("."),
                size_bytes=size_bytes,
                ingested_at=ingested_at,
            )
            prepared.append(Document(page_content=doc.page_content, metadata=meta))

        return self._finalize_chunks(prepared, filename)

    def _finalize_chunks(self, prepared: list[Document], filename: str) -> list[Document]:
        try:
            chunks = self._preprocessor.process_documents(prepared)
        except Exception as exc:
            logger.exception("Preprocessing failed for %s", filename)
            raise IngestionError("The document could not be processed. Check the server logs.") from exc

        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise EmptyDocumentError("The document produced no text chunks after cleaning.")

        total = len(chunks)
        final: list[Document] = []
        for index, chunk in enumerate(chunks):
            meta = dict(chunk.metadata)
            meta["chunk_index"] = index
            meta["chunk_total"] = total
            section = " > ".join(str(meta[h]) for h in ("h1", "h2", "h3") if meta.get(h))
            if section:
                meta["section"] = section
            final.append(Document(page_content=chunk.page_content, metadata=meta))
        return final

    def _store_original(self, path: Path, doc_id: str, extension: str) -> None:
        target = self._upload_dir / f"{doc_id}{extension}"
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)

    def _remove_orphaned_originals(self, keep_doc_id: str) -> None:
        # Stored originals are named by doc_id; orphans are files whose doc_id left the index.
        live = {meta.get("doc_id") for _cid, meta in self._db.get_metadatas() if meta.get("doc_id")}
        for stored in self._upload_dir.iterdir():
            if _STORED_NAME.match(stored.stem) and stored.stem not in live and stored.stem != keep_doc_id:
                stored.unlink(missing_ok=True)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
