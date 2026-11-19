"""Reproducible stage-by-stage benchmark.

    python scripts/benchmark.py                       # real embedding model from .env / defaults
    python scripts/benchmark.py --fake-embeddings     # isolates Chroma / BM25 / RRF overhead

Stages are timed separately: startup (embedding model init, vector store init, BM25 build,
warm-up), query embedding, dense, BM25, hybrid retrieval, and (if OPENROUTER_API_KEY is set)
end-to-end answer latency. The synthetic corpus is generated deterministically from a seed.
Numbers depend on your machine; with --fake-embeddings they say NOTHING about real model speed.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from teamdocs.core.config import Settings  # noqa: E402
from teamdocs.services.container import build_resources  # noqa: E402

VOCAB = (
    "retry backoff timeout cache index queue worker gateway token auth session deploy rollout "
    "kubernetes container image registry metric alert latency throughput schema migration "
    "database replica shard consistency lock mutex thread pool stream batch pipeline config secret"
).split()


def make_corpus(n: int, seed: int = 7) -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        words = [rng.choice(VOCAB) for _ in range(120)]
        docs.append(
            Document(
                page_content=" ".join(words) + f" chunk{i}",
                metadata={"source": f"doc{i % 20}.md", "filename": f"doc{i % 20}.md", "doc_id": f"d{i % 20}"},
            )
        )
    return docs


def timed(fn, repeat: int) -> dict[str, float]:
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "min_ms": round(samples[0], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chunks", type=int, default=2000, help="synthetic corpus size")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--fake-embeddings", action="store_true")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    embeddings = None
    if args.fake_embeddings:
        from tests.helpers import FakeEmbeddings

        embeddings = FakeEmbeddings()

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            data_dir=Path(tmp) / "data",
            model_cache_dir=Settings().model_cache_dir if not args.fake_embeddings else Path(tmp) / "m",
            collection_name="benchmark",
        )
        res = build_resources(settings, embeddings=embeddings)
        report: dict = {
            "embeddings": "FAKE (hashing)" if args.fake_embeddings else settings.embedding_model,
            "chunks": args.chunks,
            "startup_ms": res.startup_ms,
        }

        corpus = make_corpus(args.chunks)
        t = time.perf_counter()
        res.vectordb.add_documents(corpus, ids=[f"c{i}" for i in range(len(corpus))])
        res.vectordb.refresh_bm25()
        report["bulk_index_ms"] = round((time.perf_counter() - t) * 1000, 1)

        q = "retry backoff timeout for the worker queue"
        db = res.vectordb
        report["query_embedding"] = timed(lambda: db.embed_query(q), args.repeat)
        report["dense"] = timed(lambda: db.similarity_search_with_scores(q, 5), args.repeat)
        report["bm25"] = timed(lambda: db.bm25_search_with_scores(q, 5), args.repeat)
        report["bm25_filtered"] = timed(
            lambda: db.bm25_search_with_scores(q, 5, {"filename": "doc3.md"}), args.repeat
        )
        report["hybrid"] = timed(lambda: db.hybrid_search_with_scores(q, 5, 20), args.repeat)
        if res.chat_model is not None:
            report["end_to_end_with_llm"] = timed(lambda: res.rag.answer(q), 3)
        else:
            report["end_to_end_with_llm"] = "skipped: OPENROUTER_API_KEY not set"

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
