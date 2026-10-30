import hashlib
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

VECTORSTORE_PATH = Path(__file__).parent.parent.parent.parent / "vector_database"


class VectorDatabase:
    def __init__(
            self,
            persist_directory: str | Path = VECTORSTORE_PATH,
            collection_name: str = "teamdocs-ai",
            rrf_k: int = 30,
            dense_weight: float = 0.7,
            bm25_weight: float = 0.3,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
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
        self.bm25_retriever = None
        self.bm25_docs: list[Document] = []
        self.bm25_dirty = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _make_id(doc: Document) -> str:
        raw = f"{doc.metadata.get('source', '')}::{doc.page_content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_filter(metadata_filter: dict | None) -> dict | None:
        if not metadata_filter:
            return None
        cleaned = {
            key: value
            for key, value in metadata_filter.items()
            if value is not None
        }
        return cleaned if cleaned else None

    @staticmethod
    def _to_chroma_filter(metadata_filter: dict | None) -> dict | None:
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            return metadata_filter
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    @staticmethod
    def _matches_filter(
            doc: Document,
            metadata_filter: dict | None,
    ) -> bool:
        if not metadata_filter:
            return True

        return all(
            doc.metadata.get(key) == value
            for key, value in metadata_filter.items()
        )

    def _ensure_bm25(self) -> None:
        if not self.bm25_dirty:
            return

        data = self.store.get(include=["documents", "metadatas"])

        self.bm25_docs = [
            Document(
                page_content=text,
                metadata=metadata or {},
            )
            for text, metadata in zip(
                data["documents"] or [],
                data["metadatas"] or [],
            )
        ]

        if self.bm25_docs:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.bm25_docs,
                preprocess_func=self._tokenize,
            )
        else:
            self.bm25_retriever = None

        self.bm25_dirty = False

    @staticmethod
    def _clean_metadata(doc: Document) -> Document:
        cleaned = {}
        for key, value in (doc.metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                if len(value) == 0:
                    continue
                cleaned[key] = list(value)
            else:
                cleaned[key] = value
        return Document(page_content=doc.page_content, metadata=cleaned)

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        unique_docs: dict[str, Document] = {}
        for doc in documents:
            doc = self._clean_metadata(doc)
            unique_docs[self._make_id(doc)] = doc

        self.store.add_documents(
            documents=list(unique_docs.values()),
            ids=list(unique_docs.keys()),
        )

        self.bm25_dirty = True

    def delete_by_source(self, source: str) -> None:
        data = self.store.get(
            where={"source": source},
        )

        if data["ids"]:
            self.store.delete(ids=data["ids"])
            self.bm25_dirty = True

    def count(self) -> int:
        return self.store._collection.count()

    def similarity_search(
            self,
            query: str,
            k: int = 5,
            metadata_filter: dict | None = None,
    ) -> list[Document]:
        metadata_filter = self._normalize_filter(metadata_filter)

        return self.store.similarity_search(
            query=query,
            k=k,
            filter=self._to_chroma_filter(metadata_filter),
        )

    def bm25_search(
            self,
            query: str,
            k: int = 5,
            metadata_filter: dict | None = None,
    ) -> list[Document]:
        metadata_filter = self._normalize_filter(metadata_filter)

        self._ensure_bm25()

        if not self.bm25_docs:
            return []

        filtered_docs = [
            doc
            for doc in self.bm25_docs
            if self._matches_filter(doc, metadata_filter)
        ]

        if not filtered_docs:
            return []

        if metadata_filter:
            retriever = BM25Retriever.from_documents(
                filtered_docs,
                preprocess_func=self._tokenize,
            )
        else:
            retriever = self.bm25_retriever

        retriever.k = k
        results = retriever.invoke(query)

        # query_tokens = set(self._tokenize(query))
        # return [
        #     doc
        #     for doc in results
        #     if query_tokens & set(self._tokenize(doc.page_content))
        # ]
        return results

    def hybrid_search(
            self,
            query: str,
            k: int = 5,
            fetch_k: int = 20,
            metadata_filter: dict | None = None,
            dense_weight: float | None = None,
    ) -> list[Document]:
        fetch_k = max(fetch_k, k)
        metadata_filter = self._normalize_filter(metadata_filter)

        if dense_weight is None:
            dense_w = self.dense_weight
            bm25_w = self.bm25_weight
        else:
            dense_w = dense_weight
            bm25_w = 1.0 - dense_weight

        dense_docs = self.similarity_search(
            query=query,
            k=fetch_k,
            metadata_filter=metadata_filter,
        )

        bm25_docs = self.bm25_search(
            query=query,
            k=fetch_k,
            metadata_filter=metadata_filter,
        )

        fused_scores = {}
        docs_by_id = {}

        for ranked_docs, weight in (
                (dense_docs, dense_w),
                (bm25_docs, bm25_w),
        ):
            for rank, doc in enumerate(ranked_docs, start=1):
                doc_id = self._make_id(doc)

                docs_by_id[doc_id] = doc

                fused_scores[doc_id] = (
                        fused_scores.get(doc_id, 0.0)
                        + weight * (1.0 / (self.rrf_k + rank))
                )

        top_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:k]

        return [docs_by_id[doc_id] for doc_id in top_ids]
