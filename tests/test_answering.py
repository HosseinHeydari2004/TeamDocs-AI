from __future__ import annotations

import pytest

from teamdocs.services.answering import INSUFFICIENT_EVIDENCE_MESSAGE, RagService, translate_llm_error
from teamdocs.services.errors import (
    InvalidQueryError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

from .helpers import ScriptedLLM

RETRY = "# Retries\n\nThe client retries failed requests using exponential backoff with jitter.\n"
DEPLOY = "# Deploy\n\nDeployments use a Kubernetes rolling update strategy.\n"


@pytest.fixture
def loaded(resources, tmp_path):
    for name, text in (("retry.md", RETRY), ("deploy.md", DEPLOY)):
        p = tmp_path / name
        p.write_text(text)
        resources.ingestion.ingest_file(p, name)
    return resources


def test_answer_is_grounded_and_citations_map_to_the_real_source(loaded):
    llm = ScriptedLLM(lambda msgs: "Requests are retried with exponential backoff [1].")
    loaded.rag._llm = llm
    result = loaded.rag.answer("How does the retry backoff work?", top_k=2)
    first = result.sources[0]
    assert (
        first.filename == "retry.md" and first.section == "Retries" and "exponential backoff" in first.excerpt
    )
    assert first.cited and result.generated and not result.insufficient_evidence
    prompt = llm.calls[0][1].content
    assert "[1] source: retry.md" in prompt and "How does the retry backoff work?" in prompt


def test_fabricated_citation_numbers_are_stripped(loaded):
    loaded.rag._llm = ScriptedLLM("Backoff is used [1]. Also something [7][99].")
    result = loaded.rag.answer("retry backoff", top_k=2)
    assert "[7]" not in result.answer and "[99]" not in result.answer and "[1]" in result.answer
    assert [s.cited for s in result.sources] == [True, False][: len(result.sources)]


def test_not_found_token_becomes_insufficient_evidence(loaded):
    loaded.rag._llm = ScriptedLLM("NOT_FOUND")
    result = loaded.rag.answer("What is the CEO's favourite colour?")
    assert result.insufficient_evidence and result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert not any(s.cited for s in result.sources)


def test_empty_index_skips_the_llm_entirely(resources, llm):
    result = resources.rag.answer("anything at all")
    assert result.insufficient_evidence and result.sources == [] and llm.calls == []


def test_generate_false_returns_sources_without_calling_llm(loaded, llm):
    result = loaded.rag.answer("kubernetes rolling update", generate=False)
    assert result.answer is None and not result.generated and llm.calls == []
    assert result.sources[0].filename == "deploy.md"


def test_filename_filter_limits_sources(loaded):
    result = loaded.rag.answer("strategy backoff", generate=False, metadata_filter={"filename": "deploy.md"})
    assert {s.filename for s in result.sources} == {"deploy.md"}


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_every_retrieval_mode_works(loaded, mode):
    result = loaded.rag.answer("exponential backoff", mode=mode, generate=False)
    assert result.mode == mode and result.sources[0].filename == "retry.md"


def test_blank_question_and_bad_filter_are_invalid_queries(loaded):
    with pytest.raises(InvalidQueryError):
        loaded.rag.answer("   ")
    with pytest.raises(InvalidQueryError):
        loaded.rag.answer("retry", metadata_filter={"filename": {"$ne": "x"}}, generate=False)


def test_missing_llm_is_a_clear_error_but_retrieval_still_works(loaded):
    service = RagService(loaded.vectordb, None, loaded.settings)
    with pytest.raises(LLMNotConfiguredError):
        service.answer("retry")
    assert service.answer("retry backoff", generate=False).sources


def test_context_respects_the_character_budget(loaded):
    loaded.settings.__dict__["max_context_chars"] = 500
    big = "x" * 5000
    from langchain_core.documents import Document

    from rag.vector_db.vector_database import RetrievedChunk

    chunks = [
        RetrievedChunk(f"c{i}", Document(page_content=big, metadata={"filename": f"f{i}"}), 0.0)
        for i in range(4)
    ]
    assert len(loaded.rag._build_context(chunks)) < 900


def test_evidence_threshold_drops_weak_dense_only_hits(loaded):
    loaded.settings.__dict__["min_dense_similarity"] = 0.99
    assert loaded.rag.answer("zzz unrelated", mode="dense", generate=False).sources == []
    assert loaded.rag.answer("exponential backoff", generate=False).sources  # lexical match still counts


class _Err(Exception):
    def __init__(self, msg="", status=None):
        super().__init__(msg)
        self.status_code = status


@pytest.mark.parametrize(
    "exc,expected",
    [
        (TimeoutError("x"), LLMTimeoutError),
        (_Err("Request timed out"), LLMTimeoutError),
        (_Err("boom", 429), LLMRateLimitError),
        (_Err("Rate limit exceeded"), LLMRateLimitError),
        (_Err("bad key sk-or-SECRET", 401), LLMProviderError),
        (RuntimeError("kaboom with key sk-or-SECRET"), LLMProviderError),
    ],
)
def test_llm_errors_are_translated_and_never_leak_details(exc, expected):
    translated = translate_llm_error(exc)
    assert isinstance(translated, expected) and "SECRET" not in translated.message


def test_llm_failure_propagates_as_domain_error(loaded):
    loaded.rag._llm = ScriptedLLM(error=TimeoutError("slow"))
    with pytest.raises(LLMTimeoutError):
        loaded.rag.answer("retry backoff")
