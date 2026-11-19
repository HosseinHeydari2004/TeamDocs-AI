from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.vector_db.vector_database import InvalidFilterError, RetrievedChunk, VectorDatabase

from .helpers import FakeEmbeddings

DOCS = [
    Document(
        page_content="The retry policy uses exponential backoff with max_retries set to 5.",
        metadata={"source": "retry.md", "filename": "retry.md", "page_number": 2},
    ),
    Document(
        page_content="Deployment runs on Kubernetes with a rolling update strategy.",
        metadata={"source": "deploy.md", "filename": "deploy.md"},
    ),
    Document(
        page_content="Authentication uses OAuth2 bearer tokens issued by the identity service.",
        metadata={"source": "auth.md", "filename": "auth.md"},
    ),
]


@pytest.fixture
def populated(vectordb):
    vectordb.add_documents(DOCS, ids=["retry:0", "deploy:0", "auth:0"])
    return vectordb


def chunk(cid: str, text: str = "x") -> RetrievedChunk:
    return RetrievedChunk(cid, Document(page_content=f"{text} {cid}"), 0.0)


# ---------------------------------------------------------------- independent methods
def test_dense_retrieval_ranks_semantically_closest_first(populated):
    results = populated.similarity_search_with_scores("exponential backoff retry", 3)
    assert results[0].id == "retry:0"
    assert results[0].dense_score == pytest.approx(0.52, abs=0.05)  # cosine similarity, not distance
    assert all(r.bm25_score is None for r in results)


def test_bm25_retrieval_matches_exact_identifier(populated):
    results = populated.bm25_search_with_scores("max_retries", 3)
    assert [r.id for r in results] == ["retry:0"]
    assert results[0].bm25_score is not None and results[0].dense_score is None


def test_bm25_returns_nothing_when_no_query_term_matches(populated):
    assert populated.bm25_search_with_scores("zzzz qqqq", 3) == []


def test_bm25_ignores_stopword_only_queries(populated):
    assert populated.bm25_search_with_scores("what is the", 3) == []


def test_empty_database_returns_empty_results(vectordb):
    assert vectordb.hybrid_search_with_scores("anything", 3) == []
    assert vectordb.bm25_search_with_scores("anything", 3) == []
    assert vectordb.count() == 0


def test_blank_query_and_nonpositive_k_return_empty(populated):
    assert populated.similarity_search_with_scores("   ", 3) == []
    assert populated.hybrid_search_with_scores("retry", 0) == []


def test_hybrid_combines_both_signals(populated):
    results = populated.hybrid_search_with_scores("kubernetes rolling update", 3)
    assert results[0].id == "deploy:0"
    assert results[0].dense_score is not None and results[0].bm25_score is not None


# ---------------------------------------------------------------- Reciprocal Rank Fusion
def test_rrf_scores_match_the_formula():
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("b"), chunk("d")]
    fused = VectorDatabase.reciprocal_rank_fusion([(dense, 0.7), (sparse, 0.3)], rrf_k=30, k=10)
    scores = {c.id: c.score for c in fused}
    assert scores["a"] == pytest.approx(0.7 / 31)
    assert scores["b"] == pytest.approx(0.7 / 32 + 0.3 / 31)  # appears in both lists
    assert scores["c"] == pytest.approx(0.7 / 33)
    assert scores["d"] == pytest.approx(0.3 / 32)
    assert [c.id for c in fused] == ["b", "a", "c", "d"]


def test_rrf_respects_k_and_zero_weight():
    fused = VectorDatabase.reciprocal_rank_fusion(
        [([chunk("a"), chunk("b")], 1.0), ([chunk("z")], 0.0)], rrf_k=60, k=1
    )
    assert [c.id for c in fused] == ["a"]


def test_rrf_collapses_identical_text_from_different_chunks():
    twin_1 = RetrievedChunk("one", Document(page_content="Same   paragraph."), 0.0)
    twin_2 = RetrievedChunk("two", Document(page_content="same paragraph."), 0.0)
    fused = VectorDatabase.reciprocal_rank_fusion([([twin_1, twin_2, chunk("x")], 1.0)], rrf_k=30, k=5)
    assert [c.id for c in fused] == ["one", "x"]


def test_rrf_keeps_scores_from_each_retriever():
    dense = [RetrievedChunk("a", Document(page_content="a"), 0.9, dense_score=0.9)]
    sparse = [RetrievedChunk("a", Document(page_content="a"), 3.2, bm25_score=3.2)]
    fused = VectorDatabase.reciprocal_rank_fusion([(dense, 0.5), (sparse, 0.5)], rrf_k=30, k=3)
    assert fused[0].dense_score == 0.9 and fused[0].bm25_score == 3.2


def test_invalid_hybrid_weight_is_rejected(populated):
    with pytest.raises(ValueError):
        populated.hybrid_search_with_scores("retry", 3, dense_weight=1.5)


# ---------------------------------------------------------------- metadata filtering
@pytest.mark.parametrize(
    "method", ["similarity_search_with_scores", "bm25_search_with_scores", "hybrid_search_with_scores"]
)
def test_filter_is_applied_by_every_retriever(populated, method):
    results = getattr(populated, method)(
        "policy service deployment tokens", 5, metadata_filter={"filename": "auth.md"}
    )
    assert results and {r.document.metadata["filename"] for r in results} == {"auth.md"}


def test_filter_with_list_matches_any(populated):
    results = populated.hybrid_search_with_scores(
        "policy service deployment", 5, metadata_filter={"filename": ["auth.md", "deploy.md"]}
    )
    assert {r.document.metadata["filename"] for r in results} <= {"auth.md", "deploy.md"}


def test_multiple_filter_keys_are_anded(populated):
    results = populated.bm25_search_with_scores(
        "retry backoff", 5, metadata_filter={"filename": "retry.md", "page_number": 2}
    )
    assert [r.id for r in results] == ["retry:0"]
    assert (
        populated.bm25_search_with_scores(
            "retry backoff", 5, metadata_filter={"filename": "retry.md", "page_number": 9}
        )
        == []
    )


def test_none_filter_values_are_ignored_and_empty_filter_is_no_filter(populated):
    assert populated.hybrid_search_with_scores("retry", 3, metadata_filter={"filename": None})
    assert populated.hybrid_search_with_scores("retry", 3, metadata_filter={})


@pytest.mark.parametrize("bad", [{"filename": {"$ne": "x"}}, {"filename": []}, {"": "x"}, {"a": object()}])
def test_invalid_filters_raise_a_clear_error(populated, bad):
    with pytest.raises(InvalidFilterError):
        populated.hybrid_search_with_scores("retry", 3, metadata_filter=bad)


# ---------------------------------------------------------------- writes, dedupe, persistence
def test_same_chunk_id_is_upserted_not_duplicated(populated):
    populated.add_documents([DOCS[0]], ids=["retry:0"])
    assert populated.count() == 3


def test_blank_chunks_are_not_stored(vectordb):
    assert vectordb.add_documents([Document(page_content="   ", metadata={})]) == 0
    assert vectordb.count() == 0


def test_metadata_is_sanitised_for_chroma(vectordb):
    doc = Document(
        page_content="hello world",
        metadata={"source": "s", "tags": ["a", "b"], "nested": {"x": 1}, "none": None, "n": 3},
    )
    vectordb.add_documents([doc], ids=["m:0"])
    stored = vectordb.get_chunks({"source": "s"})[0].metadata
    assert stored == {"source": "s", "tags": "a, b", "n": 3}


def test_data_persists_across_instances(populated, tmp_path):
    reopened = VectorDatabase(
        persist_directory=tmp_path / "chroma", collection_name="tests-vdb", embeddings=FakeEmbeddings()
    )
    assert reopened.count() == 3
    assert [c.id for c in reopened.bm25_search_with_scores("oauth2", 2)] == ["auth:0"]
    assert reopened.similarity_search_with_scores("kubernetes", 1)[0].id == "deploy:0"


def test_delete_removes_from_dense_and_bm25(populated):
    assert populated.delete_where({"filename": "auth.md"}) == 1
    assert populated.count() == 2
    assert populated.bm25_search_with_scores("oauth2", 3) == []
    assert all(r.id != "auth:0" for r in populated.similarity_search_with_scores("oauth2 tokens", 3))


def test_delete_where_requires_a_filter(populated):
    with pytest.raises(InvalidFilterError):
        populated.delete_where({})


# ---------------------------------------------------------------- resource reuse / no rebuilds
def test_bm25_index_is_built_once_until_the_collection_changes(populated):
    populated.refresh_bm25(force=True)
    index = populated._bm25
    for query in ("retry", "kubernetes", "oauth2"):
        populated.bm25_search_with_scores(query, 3)
        populated.bm25_search_with_scores(query, 3, metadata_filter={"filename": "auth.md"})
    assert populated._bm25 is index  # filtered and unfiltered queries reuse the same index

    populated.add_documents([Document(page_content="new chunk about caching", metadata={})], ids=["new:0"])
    assert populated._bm25_dirty
    assert populated.bm25_search_with_scores("caching", 3)[0].id == "new:0"
    assert populated._bm25 is not index  # rebuilt exactly because the data changed


def test_queries_embed_only_the_query_never_the_documents(populated, embeddings):
    embedded_before = embeddings.documents_embedded
    for _ in range(5):
        populated.hybrid_search_with_scores("retry policy", 3)
    assert embeddings.documents_embedded == embedded_before
    assert embeddings.query_calls == 5  # one query embedding per hybrid search
