"""Regression: documents with more chunks than Chroma's max batch size must be indexed, not rejected."""
from langchain_core.documents import Document

from rag.vector_db.vector_database import VectorDatabase


def test_more_chunks_than_chroma_batch_limit_are_indexed(tmp_path, embeddings):
    db = VectorDatabase(collection_name="batching", persist_directory=str(tmp_path / "db"), embeddings=embeddings)
    docs = [Document(page_content=f"chunk number {i} about retries", metadata={"filename": "big.md", "doc_id": "a" * 16}) for i in range(5600)]
    assert db.add_documents(docs, ids=[f"{'a' * 16}:{i:05d}" for i in range(5600)]) == 5600
    assert db.count() == 5600
