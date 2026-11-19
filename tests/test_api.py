from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from teamdocs.api.app import create_app
from teamdocs.core.config import Settings
from teamdocs.services.container import build_resources

from .helpers import FakeEmbeddings, make_pdf

RETRY = b"# Retries\n\nThe client retries failed requests using exponential backoff with jitter.\n"


def upload(client, name="retry.md", content=RETRY):
    return client.post(
        "/api/v1/documents", files={"file": (name, io.BytesIO(content), "application/octet-stream")}
    )


# ------------------------------------------------------------------ health / readiness
def test_health_is_always_ok(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ready_reports_resources_and_counts(client):
    body = client.get("/ready").json()
    assert client.get("/ready").status_code == 200 and body["ready"] is True
    assert body["checks"]["vector_store"] and body["checks"]["llm_configured"] and body["chunks"] == 0
    assert "embedding_model_init" in body["startup_ms"]


def test_not_ready_when_startup_fails_but_process_stays_up(settings):
    def broken(_):
        raise RuntimeError("model download failed")

    with TestClient(create_app(settings, resource_factory=broken)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503 and ready.json()["ready"] is False
        assert "model download failed" in ready.json()["checks"]["error"]
        assert client.post("/api/v1/query", json={"question": "hi"}).status_code == 503


def test_ready_without_llm_still_ready(client_no_llm):
    body = client_no_llm.get("/ready").json()
    assert body["ready"] and body["checks"]["llm_configured"] is False


# ------------------------------------------------------------------ query validation / errors
def test_query_validation_errors_use_the_common_error_shape(client):
    for payload in (
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x", "top_k": 0},
        {"question": "x", "top_k": 99},
        {"question": "x", "mode": "magic"},
        {"question": "x" * 2001},
    ):
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 422, payload
        assert response.json()["error"]["code"] == "validation_error"


def test_query_returns_answer_with_sources_after_upload(client, llm):
    assert upload(client).json()["status"] == "indexed"
    llm.reply = "Failed requests are retried with exponential backoff and jitter [1]."
    body = client.post("/api/v1/query", json={"question": "How are failed requests retried?"}).json()
    assert body["generated"] and not body["insufficient_evidence"] and "[1]" in body["answer"]
    source = body["sources"][0]
    assert source["filename"] == "retry.md" and source["section"] == "Retries" and source["cited"]
    assert "exponential backoff" in source["excerpt"] and source["dense_score"] is not None
    assert body["timings_ms"]["retrieval"] >= 0


def test_query_without_openrouter_key_falls_back_to_ollama_and_reports_it_when_unreachable(client_no_llm):
    # No key -> policy "auto" selects local Ollama. Nothing listens on the (unused) port, so the
    # request must fail with a precise, actionable error - not with "key not configured".
    client_no_llm.app.state.resources.models.reload(
        client_no_llm.app.state.settings.model_copy(update={"ollama_base_url": "http://127.0.0.1:9"})
    )
    upload(client_no_llm)
    denied = client_no_llm.post("/api/v1/query", json={"question": "retry backoff"})
    assert denied.status_code == 502 and denied.json()["error"]["code"] == "llm_unreachable"
    assert "ollama" in denied.json()["error"]["message"].lower()
    ok = client_no_llm.post("/api/v1/query", json={"question": "retry backoff", "generate": False}).json()
    assert ok["answer"] is None and ok["sources"][0]["filename"] == "retry.md"


def test_llm_provider_errors_map_to_http_codes_without_leaking(client, llm):
    upload(client)
    for error, status in ((TimeoutError("t"), 504), (RuntimeError("secret sk-or-123"), 502)):
        llm.error = error
        response = client.post("/api/v1/query", json={"question": "retry backoff"})
        assert response.status_code == status and "sk-or" not in response.text


def test_unhandled_exception_returns_generic_500_without_traceback(settings, embeddings, llm):
    app = create_app(
        settings, resource_factory=lambda s: build_resources(s, embeddings=embeddings, chat_model=llm)
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.app.state.resources.rag.answer = lambda *a, **k: 1 / 0
        response = client.post("/api/v1/query", json={"question": "hi"})
    assert (
        response.status_code == 500
        and "ZeroDivision" not in response.text
        and "Traceback" not in response.text
    )
    assert response.json()["error"]["code"] == "internal_error"


# ------------------------------------------------------------------ documents
def test_upload_list_reupload_delete_flow(client):
    first = upload(client).json()
    assert first["status"] == "indexed" and first["chunk_count"] >= 1
    assert upload(client).json()["status"] == "unchanged"
    listing = client.get("/api/v1/documents").json()
    assert listing["total_documents"] == 1 and listing["documents"][0]["filename"] == "retry.md"
    assert listing["total_chunks"] == first["chunk_count"]
    assert client.post(f"/api/v1/documents/{first['doc_id']}/reindex").json()["status"] == "indexed"
    assert client.delete(f"/api/v1/documents/{first['doc_id']}").json()["deleted_chunks"] >= 1
    assert client.get("/api/v1/documents").json()["total_documents"] == 0
    assert client.delete(f"/api/v1/documents/{first['doc_id']}").status_code == 404


def test_upload_rejects_bad_types_empty_oversized_and_fake_pdfs(client, settings):
    assert upload(client, "run.exe", b"MZ...").status_code == 415
    assert upload(client, "noext", b"hello").status_code == 415
    assert upload(client, "empty.txt", b"").status_code == 422
    assert upload(client, "fake.pdf", b"this is not a pdf").status_code == 422
    big = b"a" * (settings.max_upload_bytes + 1)
    response = upload(client, "big.txt", big)
    assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"
    assert client.get("/api/v1/documents").json()["total_documents"] == 0
    assert list(settings.tmp_dir.iterdir()) == []  # no temp files left behind


def test_upload_sanitises_traversal_filenames(client):
    body = upload(client, "../../etc/evil.md", RETRY).json()
    assert body["filename"] == "evil.md"


def test_pdf_upload_citation_has_the_right_page(client):
    pdf = make_pdf(["Intro about alpha systems", "Rotation of secrets happens every ninety days"])
    assert upload(client, "security.pdf", pdf).status_code == 200
    body = client.post("/api/v1/query", json={"question": "secrets rotation", "generate": False}).json()
    assert body["sources"][0]["page"] == 2 and body["sources"][0]["filename"] == "security.pdf"


def test_concurrent_identical_uploads_are_indexed_once(client):
    with ThreadPoolExecutor(5) as pool:
        statuses = [r.json()["status"] for r in pool.map(lambda _: upload(client), range(5))]
    assert statuses.count("indexed") == 1
    assert client.get("/api/v1/documents").json()["total_documents"] == 1


# ------------------------------------------------------------------ resource reuse & persistence
def test_resources_are_created_once_and_reused_for_all_requests(settings, llm):
    created = []
    embeddings = FakeEmbeddings()

    def factory(s):
        created.append(1)
        return build_resources(s, embeddings=embeddings, chat_model=llm)

    with TestClient(create_app(settings, resource_factory=factory)) as client:
        upload(client)
        docs_embedded = embeddings.documents_embedded
        queries_before = embeddings.query_calls
        for _ in range(5):
            client.post("/api/v1/query", json={"question": "retry backoff", "mode": "dense"})
        assert len(created) == 1
        assert embeddings.documents_embedded == docs_embedded  # documents never re-embedded
        assert embeddings.query_calls - queries_before == 5  # one query embedding per request


def test_indexed_data_survives_an_application_restart(settings, llm):
    def run():
        return TestClient(
            create_app(
                settings,
                resource_factory=lambda s: build_resources(s, embeddings=FakeEmbeddings(), chat_model=llm),
            )
        )

    with run() as first:
        upload(first)
    with run() as second:
        assert second.get("/api/v1/documents").json()["total_documents"] == 1
        assert second.get("/ready").json()["chunks"] >= 1
        hit = second.post("/api/v1/query", json={"question": "exponential backoff", "generate": False}).json()
        assert hit["sources"][0]["filename"] == "retry.md"
        assert upload(second).json()["status"] == "unchanged"  # not re-indexed after restart


# ------------------------------------------------------------------ frontend + config
def test_frontend_and_runtime_config_are_served(settings, embeddings, llm):
    settings.frontend_api_base_url = "http://api.example:9000/"
    app = create_app(
        settings, resource_factory=lambda s: build_resources(s, embeddings=embeddings, chat_model=llm)
    )
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200 and "TeamDocs AI" in home.text
        assert 'apiBaseUrl: "http://api.example:9000"' in client.get("/config.js").text
        assert client.get("/openapi.json").status_code == 200


def test_unknown_api_path_uses_json_error_shape(client):
    response = client.get("/api/v1/nope")
    assert response.status_code == 404 and response.json()["error"]["code"] == "http_error"


def test_settings_object_is_isolated_from_environment(settings):
    assert isinstance(settings, Settings) and settings.openrouter_api_key is None
