"""Hybrid (dense + BM25) retrieval over a persistent Chroma collection.

Design notes
------------
* The embedding model, Chroma client/collection and BM25 index are created once
  and reused for every query. Nothing here is rebuilt per request.
* The BM25 index is built lazily from the collection and rebuilt only after the
  collection has changed (``_bm25_dirty``). Metadata filters are applied as a
  mask over the existing index, so a filtered query does not rebuild anything.
* Dense and BM25 results are fused with weighted Reciprocal Rank Fusion:
  ``score(d) = sum_i  w_i / (rrf_k + rank_i(d))``.
* All mutating operations and the BM25 rebuild are serialised by one lock.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

VECTORSTORE_PATH = Path(__file__).parent.parent.parent.parent / "vector_database"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in into is it its of on "
    "or our that the their then there these this to was we were what when where which "
    "who why will with you your".split()
)
_SCALAR = (str, int, float, bool)


class InvalidFilterError(ValueError):
    """Raised when a metadata filter is malformed."""


@dataclass(slots=True)
class RetrievedChunk:
    """A retrieved chunk together with the scores that produced its rank."""

    id: str
    document: Document
    score: float                      # score used for ordering (method specific)
    dense_score: float | None = None  # cosine similarity in [-1, 1] (1 = identical)
    bm25_score: float | None = None   # raw BM25 score (corpus dependent, not normalised)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def build_embeddings(model_name: str, cache_dir: str | Path | None, device: str | None):
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs: dict[str, Any] = {}
    if device:
        model_kwargs["device"] = device
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(cache_dir) if cache_dir else None,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


_ADD_BATCH = 4000  # below Chroma's max batch size


class VectorDatabase:
    def __init__(
            self,
            persist_directory: str | Path = VECTORSTORE_PATH,
            collection_name: str = "teamdocs-ai",
            rrf_k: int = 30,
            dense_weight: float = 0.7,
            bm25_weight: float = 0.3,
            *,
            embeddings: Embeddings | None = None,
            embedding_model: str = DEFAULT_EMBEDDING_MODEL,
            model_cache_dir: str | Path | None = None,
            embedding_device: str | None = None,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if dense_weight < 0 or bm25_weight < 0 or (dense_weight + bm25_weight) == 0:
            raise ValueError("retrieval weights must be non-negative and not both zero")

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.embeddings = embeddings or build_embeddings(
            embedding_model, model_cache_dir, embedding_device
        )
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self._lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[Document] = []
        self._bm25_token_sets: list[frozenset[str]] = []
        self._bm25_dirty = True

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _make_id(doc: Document) -> str:
        raw = f"{doc.metadata.get('source', '')}::{doc.page_content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _content_key(doc: Document) -> str:
        return hashlib.sha1(" ".join(doc.page_content.split()).lower().encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_metadata(doc: Document) -> Document:
        """Keep only values Chroma can store (str/int/float/bool).

        Lists of scalars are joined into a comma separated string; dicts, nested
        structures and ``None`` are dropped.
        """
        cleaned: dict[str, Any] = {}
        for key, value in (doc.metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, _SCALAR):
                cleaned[str(key)] = value
            elif isinstance(value, (list, tuple, set)):
                items = [str(v) for v in value if isinstance(v, _SCALAR)]
                if items:
                    cleaned[str(key)] = ", ".join(items)
        return Document(page_content=doc.page_content, metadata=cleaned, id=doc.id)

    @staticmethod
    def _normalize_filter(metadata_filter: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Validate and normalise a filter. ``None``/empty -> no filter."""
        if not metadata_filter:
            return None
        if not isinstance(metadata_filter, Mapping):
            raise InvalidFilterError("metadata_filter must be a mapping of field -> value")
        cleaned: dict[str, Any] = {}
        for key, value in metadata_filter.items():
            if value is None:
                continue
            if not isinstance(key, str) or not key:
                raise InvalidFilterError("metadata filter keys must be non-empty strings")
            if isinstance(value, (list, tuple, set)):
                values = list(value)
                if not values or not all(isinstance(v, _SCALAR) for v in values):
                    raise InvalidFilterError(f"filter {key!r}: list values must be non-empty scalars")
                cleaned[key] = values
            elif isinstance(value, _SCALAR):
                cleaned[key] = value
            else:
                raise InvalidFilterError(f"filter {key!r}: unsupported value type {type(value).__name__}")
        return cleaned or None

    @staticmethod
    def _to_chroma_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        if not metadata_filter:
            return None
        clauses = [
            {key: {"$in": value}} if isinstance(value, list) else {key: value}
            for key, value in metadata_filter.items()
        ]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _matches_filter(metadata: Mapping[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            actual = metadata.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    # --------------------------------------------------------------- BM25 index
    def refresh_bm25(self, force: bool = False) -> int:
        """(Re)build the BM25 index if the collection changed. Returns indexed chunks."""
        with self._lock:
            if not self._bm25_dirty and not force:
                return len(self._bm25_ids)

            data = self.store.get(include=["documents", "metadatas"])
            ids = data.get("ids") or []
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []

            self._bm25_ids = list(ids)
            self._bm25_docs = [
                Document(id=i, page_content=t or "", metadata=m or {})
                for i, t, m in zip(ids, texts, metas)
            ]
            tokenized = [_tokenize(d.page_content) for d in self._bm25_docs]
            self._bm25_token_sets = [frozenset(t) for t in tokenized]
            # BM25Okapi cannot be built from an empty corpus or all-empty documents.
            self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None
            self._bm25_dirty = False
            logger.info("BM25 index built: %d chunks", len(self._bm25_ids))
            return len(self._bm25_ids)

    # ------------------------------------------------------------------ writes
    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> int:
        """Embed and store chunks. Chunks with an identical id are upserted, not duplicated."""
        if not documents:
            return 0
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")

        unique: dict[str, Document] = {}
        for i, doc in enumerate(documents):
            if not doc.page_content or not doc.page_content.strip():
                continue
            cleaned = self._clean_metadata(doc)
            chunk_id = ids[i] if ids is not None else self._make_id(cleaned)
            unique[chunk_id] = cleaned

        if not unique:
            return 0

        items = list(unique.items())
        with self._lock:
            # Chroma rejects a single call above its max batch size (5461 here): a large PDF or web page
            # can produce more chunks than that, so write in fixed-size batches.
            for start in range(0, len(items), _ADD_BATCH):
                batch = items[start:start + _ADD_BATCH]
                self.store.add_documents(documents=[d for _, d in batch], ids=[i for i, _ in batch])
            self._bm25_dirty = True
        return len(unique)

    def delete_ids(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        with self._lock:
            self.store.delete(ids=ids)
            self._bm25_dirty = True
        return len(ids)

    def delete_where(self, where: Mapping[str, Any]) -> int:
        normalized = self._normalize_filter(where)
        if not normalized:
            raise InvalidFilterError("delete_where requires a non-empty filter")
        with self._lock:
            data = self.store.get(where=self._to_chroma_filter(normalized))
            return self.delete_ids(data.get("ids") or [])

    def delete_by_source(self, source: str) -> int:
        return self.delete_where({"source": source})

    # ------------------------------------------------------------------- reads
    def count(self) -> int:
        return self.store._collection.count()

    def get_metadatas(self, where: Mapping[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
        normalized = self._normalize_filter(where)
        data = self.store.get(where=self._to_chroma_filter(normalized), include=["metadatas"])
        return list(zip(data.get("ids") or [], [m or {} for m in (data.get("metadatas") or [])]))

    def get_chunks(self, where: Mapping[str, Any] | None = None, limit: int | None = None) -> list[Document]:
        normalized = self._normalize_filter(where)
        data = self.store.get(
            where=self._to_chroma_filter(normalized),
            include=["documents", "metadatas"],
            limit=limit,
        )
        return [
            Document(id=i, page_content=t or "", metadata=m or {})
            for i, t, m in zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or [])
        ]

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    # ------------------------------------------------------------------ dense
    def similarity_search_with_scores(
            self,
            query: str,
            k: int = 5,
            metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if k <= 0 or not query or not query.strip():
            return []
        normalized = self._normalize_filter(metadata_filter)
        pairs = self.store.similarity_search_with_score(
            query=query, k=k, filter=self._to_chroma_filter(normalized)
        )
        results: list[RetrievedChunk] = []
        for doc, distance in pairs:
            similarity = 1.0 - float(distance)  # collection uses cosine distance
            chunk_id = doc.id or self._make_id(doc)
            results.append(RetrievedChunk(chunk_id, doc, similarity, dense_score=similarity))
        return results

    # ------------------------------------------------------------------- BM25
    def bm25_search_with_scores(
            self,
            query: str,
            k: int = 5,
            metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        normalized = self._normalize_filter(metadata_filter)
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        self.refresh_bm25()
        with self._lock:
            bm25, docs, ids, token_sets = (
                self._bm25, self._bm25_docs, self._bm25_ids, self._bm25_token_sets,
            )
        if bm25 is None or not docs:
            return []

        scores = bm25.get_scores(query_tokens)
        query_set = set(query_tokens)

        candidates: list[tuple[float, int, int]] = []
        for index, doc in enumerate(docs):
            overlap = len(query_set & token_sets[index])
            # BM25Okapi IDF can be <= 0 on tiny corpora, so "matches the query" is
            # decided by token overlap rather than by the score being positive.
            if overlap == 0 or not self._matches_filter(doc.metadata, normalized):
                continue
            candidates.append((float(scores[index]), overlap, index))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            RetrievedChunk(ids[index], docs[index], score, bm25_score=score)
            for score, _overlap, index in candidates[:k]
        ]

    # ----------------------------------------------------------------- hybrid
    def hybrid_search_with_scores(
            self,
            query: str,
            k: int = 5,
            fetch_k: int = 20,
            metadata_filter: Mapping[str, Any] | None = None,
            dense_weight: float | None = None,
    ) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        fetch_k = max(fetch_k, k)
        normalized = self._normalize_filter(metadata_filter)

        if dense_weight is None:
            dense_w, bm25_w = self.dense_weight, self.bm25_weight
        else:
            if not 0.0 <= dense_weight <= 1.0:
                raise ValueError("dense_weight must be between 0 and 1")
            dense_w, bm25_w = dense_weight, 1.0 - dense_weight

        dense = self.similarity_search_with_scores(query, fetch_k, normalized)
        sparse = self.bm25_search_with_scores(query, fetch_k, normalized)
        return self.reciprocal_rank_fusion(
            [(dense, dense_w), (sparse, bm25_w)], rrf_k=self.rrf_k, k=k
        )

    @classmethod
    def reciprocal_rank_fusion(
            cls,
            ranked_lists: list[tuple[list[RetrievedChunk], float]],
            *,
            rrf_k: int,
            k: int,
    ) -> list[RetrievedChunk]:
        """Weighted RRF: ``sum_i weight_i / (rrf_k + rank_i)`` with 1-based ranks.

        Chunks are identified by chunk id; chunks with identical text (e.g. the
        same paragraph in two files) are collapsed, keeping the best ranked one.
        """
        fused: dict[str, float] = {}
        merged: dict[str, RetrievedChunk] = {}

        for ranked, weight in ranked_lists:
            if weight <= 0:
                continue
            for rank, item in enumerate(ranked, start=1):
                fused[item.id] = fused.get(item.id, 0.0) + weight / (rrf_k + rank)
                existing = merged.get(item.id)
                if existing is None:
                    merged[item.id] = RetrievedChunk(
                        item.id, item.document, 0.0, item.dense_score, item.bm25_score
                    )
                else:
                    existing.dense_score = existing.dense_score if existing.dense_score is not None else item.dense_score
                    existing.bm25_score = existing.bm25_score if existing.bm25_score is not None else item.bm25_score

        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))
        results: list[RetrievedChunk] = []
        seen_content: set[str] = set()
        for chunk_id in ordered:
            chunk = merged[chunk_id]
            chunk.score = fused[chunk_id]
            content_key = cls._content_key(chunk.document)
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            results.append(chunk)
            if len(results) >= k:
                break
        return results

    # --------------------------------------------- document-only compat API
    def similarity_search(self, query: str, k: int = 5, metadata_filter: dict | None = None) -> list[Document]:
        return [c.document for c in self.similarity_search_with_scores(query, k, metadata_filter)]

    def bm25_search(self, query: str, k: int = 5, metadata_filter: dict | None = None) -> list[Document]:
        return [c.document for c in self.bm25_search_with_scores(query, k, metadata_filter)]

    def hybrid_search(
            self,
            query: str,
            k: int = 5,
            fetch_k: int = 20,
            metadata_filter: dict | None = None,
            dense_weight: float | None = None,
    ) -> list[Document]:
        return [
            c.document
            for c in self.hybrid_search_with_scores(query, k, fetch_k, metadata_filter, dense_weight)
        ]
