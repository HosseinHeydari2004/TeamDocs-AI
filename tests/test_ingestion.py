from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from teamdocs.services.errors import DocumentNotFoundError, EmptyDocumentError, InvalidFileError

from .helpers import make_pdf

RETRY_MD = "# Retries\n\nThe client retries failed requests with exponential backoff and jitter.\n"


def write(tmp_path, name, content):
    path = tmp_path / "in" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def test_ingest_indexes_chunks_with_consistent_metadata(resources, tmp_path):
    result = resources.ingestion.ingest_file(write(tmp_path, "retry.md", RETRY_MD), "retry.md")
    assert result.status == "indexed" and result.chunk_count >= 1
    chunks = resources.vectordb.get_chunks({"doc_id": result.doc_id})
    meta = chunks[0].metadata
    assert meta["filename"] == meta["source"] == "retry.md"  # no absolute server path
    assert meta["section"] == "Retries" and meta["file_type"] == "md"
    assert meta["chunk_total"] == len(chunks) and "file_directory" not in meta
    assert str(tmp_path) not in " ".join(str(v) for v in meta.values())


def test_pdf_page_numbers_are_one_based_and_preserved(resources, tmp_path):
    path = write(
        tmp_path, "paper.pdf", make_pdf(["Alpha content on first page", "Beta content on second page"])
    )
    resources.ingestion.ingest_file(path, "paper.pdf")
    hit = resources.vectordb.bm25_search_with_scores("beta", 1)[0]
    assert hit.document.metadata["page_number"] == 2


def test_identical_file_is_not_embedded_twice(resources, embeddings, tmp_path):
    path = write(tmp_path, "retry.md", RETRY_MD)
    first = resources.ingestion.ingest_file(path, "retry.md")
    embedded = embeddings.documents_embedded
    second = resources.ingestion.ingest_file(path, "retry.md")
    assert second.status == "unchanged" and second.doc_id == first.doc_id
    assert embeddings.documents_embedded == embedded  # no re-embedding
    assert resources.vectordb.count() == first.chunk_count  # no duplicate chunks


def test_same_content_under_another_name_is_flagged_duplicate(resources, embeddings, tmp_path):
    resources.ingestion.ingest_file(write(tmp_path, "a.md", RETRY_MD), "a.md")
    embedded = embeddings.documents_embedded
    result = resources.ingestion.ingest_file(write(tmp_path, "b.md", RETRY_MD), "b.md")
    assert result.status == "duplicate" and result.filename == "a.md"
    assert embeddings.documents_embedded == embedded
    assert [d.filename for d in resources.ingestion.list_documents()] == ["a.md"]


def test_changed_content_with_same_name_replaces_the_old_version(resources, tmp_path):
    first = resources.ingestion.ingest_file(write(tmp_path, "policy.md", RETRY_MD), "policy.md")
    second = resources.ingestion.ingest_file(
        write(tmp_path, "policy.md", "# Policy\n\nSecrets are rotated every 90 days.\n"), "policy.md"
    )
    assert (
        second.status == "replaced"
        and second.doc_id != first.doc_id
        and second.replaced_chunks == first.chunk_count
    )
    docs = resources.ingestion.list_documents()
    assert len(docs) == 1 and docs[0].doc_id == second.doc_id
    assert resources.vectordb.bm25_search_with_scores("exponential backoff", 3) == []  # old text is gone
    assert resources.vectordb.bm25_search_with_scores("rotated", 3)
    stored = {p.stem for p in resources.settings.upload_dir.iterdir()}
    assert stored == {second.doc_id}  # old original removed


def test_delete_removes_chunks_bm25_entries_and_stored_file(resources, tmp_path):
    result = resources.ingestion.ingest_file(write(tmp_path, "retry.md", RETRY_MD), "retry.md")
    assert resources.ingestion.delete_document(result.doc_id) == result.chunk_count
    assert resources.vectordb.count() == 0
    assert resources.vectordb.bm25_search_with_scores("backoff", 3) == []
    assert list(resources.settings.upload_dir.iterdir()) == []
    with pytest.raises(DocumentNotFoundError):
        resources.ingestion.delete_document(result.doc_id)


def test_reindex_removes_stale_tail_chunks(resources, tmp_path):
    big = "# Big\n\n" + "\n\n".join(f"Paragraph {i} " + "filler " * 80 for i in range(12))
    result = resources.ingestion.ingest_file(write(tmp_path, "big.md", big), "big.md")
    assert result.chunk_count > 3
    resources.settings.__dict__["chunk_size"] = 4000  # not used by the already-built chunker...
    resources.ingestion._preprocessor.chunker = type(resources.ingestion._preprocessor.chunker)(4000, 100)
    again = resources.ingestion.reindex_document(result.doc_id)
    stored = resources.vectordb.get_chunks({"doc_id": result.doc_id})
    assert len(stored) == again.chunk_count < result.chunk_count
    assert {c.metadata["chunk_total"] for c in stored} == {again.chunk_count}


def test_reindex_without_stored_original_is_a_clean_error(resources, tmp_path):
    result = resources.ingestion.ingest_file(write(tmp_path, "retry.md", RETRY_MD), "retry.md")
    for f in resources.settings.upload_dir.iterdir():
        f.unlink()
    with pytest.raises(DocumentNotFoundError):
        resources.ingestion.reindex_document(result.doc_id)


def test_empty_and_whitespace_files_are_rejected(resources, tmp_path):
    with pytest.raises(EmptyDocumentError):
        resources.ingestion.ingest_file(write(tmp_path, "empty.txt", "   \n\n  "), "empty.txt")
    assert resources.vectordb.count() == 0


def test_corrupt_pdf_is_reported_as_invalid_not_as_a_crash(resources, tmp_path):
    with pytest.raises(InvalidFileError):
        resources.ingestion.ingest_file(
            write(tmp_path, "bad.pdf", b"%PDF-1.4 this is not really a pdf"), "bad.pdf"
        )
    assert resources.vectordb.count() == 0


def test_a_failing_embedder_leaves_the_index_unchanged(resources, tmp_path, monkeypatch):
    resources.ingestion.ingest_file(write(tmp_path, "keep.md", RETRY_MD), "keep.md")
    before = resources.vectordb.count()

    def boom(_texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(resources.embeddings, "embed_documents", boom)
    from teamdocs.services.errors import IngestionError

    with pytest.raises(IngestionError) as info:
        resources.ingestion.ingest_file(
            write(tmp_path, "new.md", "# New\n\nSomething else entirely.\n"), "new.md"
        )
    assert "embedding service down" not in info.value.message  # internals not leaked
    assert resources.vectordb.count() == before


def test_concurrent_uploads_of_the_same_file_index_it_once(resources, embeddings, tmp_path):
    path = write(tmp_path, "retry.md", RETRY_MD)
    barrier = threading.Barrier(6)

    def worker(_):
        barrier.wait()
        return resources.ingestion.ingest_file(path, "retry.md").status

    with ThreadPoolExecutor(6) as pool:
        statuses = list(pool.map(worker, range(6)))
    assert statuses.count("indexed") == 1 and statuses.count("unchanged") == 5
    assert embeddings.documents_embedded == resources.vectordb.count()  # every chunk embedded exactly once
    assert len(resources.ingestion.list_documents()) == 1


def test_filename_sanitising_blocks_path_traversal():
    from teamdocs.services.ingestion import safe_filename

    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\Users\\me\\..\\evil.txt") == "evil.txt"
    assert safe_filename("  re\x00port.md ") == "report.md"
    with pytest.raises(InvalidFileError):
        safe_filename("..")
    with pytest.raises(InvalidFileError):
        safe_filename(None)


def test_document_list_cache_is_invalidated_by_writes(resources, tmp_path):
    """list_documents() is cached for /ready polling; every write must be visible immediately."""
    ing = resources.ingestion
    assert ing.list_documents() == []
    first = ing.ingest_file(write(tmp_path, "a.md", "# A\n\nalpha text about retries and backoff.\n"), "a.md")
    assert [d.doc_id for d in ing.list_documents()] == [first.doc_id]
    ing.ingest_file(write(tmp_path, "b.md", "# B\n\nbeta text about deployment pipelines.\n"), "b.md")
    assert len(ing.list_documents()) == 2
    ing.delete_document(first.doc_id)
    assert [d.filename for d in ing.list_documents()] == ["b.md"]
