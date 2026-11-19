"""Loaders, cleaners and chunkers (regression tests for bugs found in the audit)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.chunker.pipeline import ChunkerPipeline
from rag.cleaners.pipelline import CleanerPipeline
from rag.loaders.pipeline import LoaderPipeline

from .helpers import make_pdf


def test_text_and_markdown_loading_sets_source_and_file_type(tmp_path):
    (tmp_path / "notes.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Title\n\nbody", encoding="utf-8")
    loader = LoaderPipeline()
    txt = loader.load_file(tmp_path / "notes.txt")
    md = loader.load_file(tmp_path / "guide.md")
    assert txt[0].metadata["file_type"] == "txt" and md[0].metadata["file_type"] == "md"
    assert md[0].page_content.startswith("# Title")  # raw markdown kept so headings survive


def test_non_utf8_text_is_decoded_instead_of_crashing(tmp_path):
    (tmp_path / "legacy.txt").write_bytes("caf\xe9 au lait".encode("latin-1"))
    assert "lait" in LoaderPipeline().load_file(tmp_path / "legacy.txt")[0].page_content


def test_pdf_loader_returns_one_document_per_page_with_page_index(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf(["First page about alpha", "Second page about beta"]))
    docs = LoaderPipeline().load_file(path)
    assert [d.metadata["page"] for d in docs] == [0, 1]
    assert "beta" in docs[1].page_content


def test_pptx_loader_returns_documents(tmp_path):
    """Regression: PPTXLoader.load() used to return None."""
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Architecture overview"
    slide.placeholders[1].text = "The gateway forwards requests to the worker pool."
    path = tmp_path / "deck.pptx"
    prs.save(path)
    docs = LoaderPipeline().load_file(path)
    assert docs and "gateway" in " ".join(d.page_content for d in docs)


def test_docx_loader_returns_text(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("The cache invalidation strategy is write-through.")
    path = tmp_path / "spec.docx"
    document.save(path)
    docs = LoaderPipeline().load_file(path)
    assert "write-through" in " ".join(d.page_content for d in docs)


def test_loader_kwargs_are_actually_passed_to_loaders():
    """Regression: per-suffix kwargs were computed and then ignored."""
    loader = LoaderPipeline(per_suffix_kwargs={".pdf": {"backend": "unstructured", "strategy": "fast"}})
    assert loader.get_loader("x.pdf").options == {"backend": "unstructured", "strategy": "fast"}


def test_unsupported_extension_is_rejected_with_clear_error(tmp_path):
    (tmp_path / "x.xyz").write_text("data")
    with pytest.raises(ValueError, match="Unsupported file format"):
        LoaderPipeline().load_file(tmp_path / "x.xyz")


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        LoaderPipeline().load_file(tmp_path / "nope.txt")


def test_cleaner_routes_web_documents_to_the_web_cleaner():
    """Regression: URL documents raised ValueError before reaching the web cleaner."""
    doc = Document(
        page_content="<p>Hello   world</p>  text",
        metadata={"source": "https://example.com/docs", "file_type": "web"},
    )
    assert CleanerPipeline().clean([doc])  # must not raise


def test_cleaner_accepts_pptx_and_reports_unsupported_types_cleanly():
    CleanerPipeline().clean(
        [Document(page_content="slide text", metadata={"source": "deck.pptx", "file_type": "pptx"})]
    )
    with pytest.raises(ValueError, match="(?i)unsupported|cannot determine"):  # was AttributeError
        CleanerPipeline().clean(
            [Document(page_content="x", metadata={"source": "weird.xyz", "file_type": "xyz"})]
        )


def test_chunker_does_not_use_semantic_document_type_for_routing():
    """Regression: LLM metadata document_type='code' used to select the code chunker
    and was then overwritten with the chunker name."""
    doc = Document(
        page_content="Plain prose. " * 200, metadata={"source": "readme.txt", "document_type": "code"}
    )
    chunks = ChunkerPipeline(chunk_size=300, overlap=30).chunk([doc])
    assert len(chunks) > 1
    assert {c.metadata["document_type"] for c in chunks} == {"code"}  # semantic value preserved
    assert {c.metadata["chunk_strategy"] for c in chunks} == {"text"}


def test_markdown_chunks_keep_heading_metadata_and_source():
    text = "# Guide\n\nIntro.\n\n## Retries\n\nExponential backoff with jitter.\n"
    doc = Document(page_content=text, metadata={"source": "guide.md", "doc_id": "abc"})
    chunks = ChunkerPipeline().chunk([doc])
    retry_chunk = next(c for c in chunks if "backoff" in c.page_content)
    assert retry_chunk.metadata["h2"] == "Retries" and retry_chunk.metadata["h1"] == "Guide"
    assert retry_chunk.metadata["source"] == "guide.md" and retry_chunk.metadata["doc_id"] == "abc"


def test_chunk_size_setting_is_honoured_for_markdown():
    doc = Document(page_content="# H\n\n" + ("word " * 600), metadata={"source": "big.md"})
    chunks = ChunkerPipeline(chunk_size=300, overlap=30).chunk([doc])
    assert max(len(c.page_content) for c in chunks) <= 300


def test_empty_documents_are_skipped_by_the_chunker():
    assert ChunkerPipeline().chunk([Document(page_content="  \n ", metadata={"source": "a.txt"})]) == []
