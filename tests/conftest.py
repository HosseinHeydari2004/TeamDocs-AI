from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.vector_db.vector_database import VectorDatabase
from teamdocs.api.app import create_app
from teamdocs.core.config import Settings
from teamdocs.services.container import build_resources

from .helpers import FakeEmbeddings, ScriptedLLM


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Isolate tests from the developer's real .env / environment."""
    for key in list(__import__("os").environ):
        if key.upper() in {f.upper() for f in Settings.model_fields}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # so a stray .env in the repo root is never read


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        collection_name="tests-collection",
    )


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def vectordb(tmp_path, embeddings) -> VectorDatabase:
    return VectorDatabase(
        persist_directory=tmp_path / "chroma", collection_name="tests-vdb", embeddings=embeddings
    )


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def resources(settings, embeddings, llm):
    return build_resources(settings, embeddings=embeddings, chat_model=llm)


@pytest.fixture
def client(settings, embeddings, llm):
    app = create_app(
        settings, resource_factory=lambda s: build_resources(s, embeddings=embeddings, chat_model=llm)
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_llm(settings, embeddings):
    app = create_app(
        settings, resource_factory=lambda s: build_resources(s, embeddings=embeddings, chat_model=None)
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def anyio_backend():
    return "asyncio"  # the agents use asyncio primitives (wait_for, ...)
