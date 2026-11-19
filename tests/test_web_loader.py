"""URL import: the existing WebLoader wired into ingestion + API, with SSRF protection.

Downloads run against a REAL local HTTP server (loopback is only allowed via WEB_ALLOW_PRIVATE_HOSTS in these
tests). The SSRF rules themselves are tested with the default (strict) configuration.
"""
from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from fastapi.testclient import TestClient

from rag.loaders.web_loader.safe_http import SafeSession, WebFetchError, validate_public_url
from rag.loaders.web_loader.web import WebLoader
from teamdocs.api.app import create_app
from teamdocs.services.container import build_resources

PAGES = {
    "/docs/retry": b"<html><head><title>x</title></head><body><nav>menu</nav><article><h1>Retry policy</h1>"
                   b"<p>Requests are retried with exponential backoff and jitter, at most five times.</p></article>"
                   b"<footer>cookie banner</footer></body></html>",
}


class _Handler(BaseHTTPRequestHandler):
    def _reply(self, status, ctype, body, extra=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802
        path = self.path
        if path == "/redirect":
            return self._reply(302, "text/plain", b"", {"Location": "/docs/retry"})
        if path == "/loop":
            return self._reply(302, "text/plain", b"", {"Location": "/loop"})
        if path == "/slow":
            time.sleep(3)
            return self._reply(200, "text/html", b"<p>late</p>")
        if path == "/pdf":
            return self._reply(200, "application/pdf", b"%PDF-1.4 binary")
        if path == "/big":
            return self._reply(200, "text/html", b"<p>" + b"a" * 200_000 + b"</p>")
        if path == "/empty":
            return self._reply(200, "text/html", b"<html><body><script>var x=1</script></body></html>")
        body = PAGES.get(path)
        if body is None:
            return self._reply(404, "text/html", b"<p>nope</p>")
        return self._reply(200, "text/html; charset=utf-8", body)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def web():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture
def client(settings, embeddings, web):
    cfg = settings.model_copy(update={"web_allow_private_hosts": True, "web_fetch_timeout_seconds": 1.0, "web_max_bytes": 100_000})
    app = create_app(cfg, resource_factory=lambda s: build_resources(s, embeddings=embeddings, chat_model=None))
    with TestClient(app) as c:
        yield c


def ingest(client, url):
    return client.post("/api/v1/documents/url", json={"url": url})


# --------------------------------------------------------------------------- happy path
def test_url_is_downloaded_indexed_and_retrievable_with_its_source_url(client, web):
    r = ingest(client, f"{web}/docs/retry")
    assert r.status_code == 200 and r.json()["status"] == "indexed" and r.json()["chunk_count"] >= 1
    assert r.json()["filename"] == f"{web}/docs/retry"
    docs = client.get("/api/v1/documents").json()["documents"]
    assert docs[0]["file_type"] == "web"
    hit = client.post("/api/v1/query", json={"question": "exponential backoff jitter", "generate": False}).json()["sources"][0]
    assert hit["source_url"] == f"{web}/docs/retry" and "exponential backoff" in hit["excerpt"]
    assert "cookie banner" not in hit["excerpt"] and "menu" not in hit["excerpt"]  # <article> extraction drops boilerplate


def test_reimport_of_unchanged_page_does_not_reembed_and_reindex_refetches(client, web, embeddings):
    ingest(client, f"{web}/docs/retry")
    embedded = embeddings.documents_embedded
    again = ingest(client, f"{web}/docs/retry").json()
    assert again["status"] == "unchanged" and embeddings.documents_embedded == embedded
    doc_id = client.get("/api/v1/documents").json()["documents"][0]["doc_id"]
    assert client.post(f"/api/v1/documents/{doc_id}/reindex").json()["status"] in ("indexed", "replaced")


def test_changed_page_replaces_the_previous_version(client, web):
    ingest(client, f"{web}/docs/retry")
    PAGES["/docs/retry"] = PAGES["/docs/retry"].replace(b"five times", b"ten times")
    try:
        r = ingest(client, f"{web}/docs/retry").json()
    finally:
        PAGES["/docs/retry"] = PAGES["/docs/retry"].replace(b"ten times", b"five times")
    assert r["status"] == "replaced" and r["replaced_chunks"] >= 1
    assert client.get("/api/v1/documents").json()["total_documents"] == 1


def test_redirects_are_followed(client, web):
    assert ingest(client, f"{web}/redirect").status_code == 200


def test_web_documents_can_be_deleted(client, web):
    doc_id = ingest(client, f"{web}/docs/retry").json()["doc_id"]
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 200
    assert client.get("/api/v1/documents").json()["total_documents"] == 0


# --------------------------------------------------------------------------- failure handling
@pytest.mark.parametrize(
    ("path", "status", "code"),
    [("/missing", 502, "url_fetch_failed"), ("/pdf", 502, "url_fetch_failed"), ("/big", 502, "url_fetch_failed"),
     ("/slow", 504, "url_timeout"), ("/loop", 502, "url_fetch_failed"), ("/empty", 422, "empty_document")],
)
def test_fetch_failures_map_to_clear_errors(client, web, path, status, code):
    r = ingest(client, f"{web}{path}")
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert client.get("/api/v1/documents").json()["total_documents"] == 0  # nothing half-indexed


def test_connection_refused_and_invalid_input(client):
    assert ingest(client, "http://127.0.0.1:9/").json()["error"]["code"] == "url_fetch_failed"
    for bad in ("not a url", "ftp://example.com/x", "javascript:alert(1)", "http://user:pw@example.com/"):
        r = ingest(client, bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_url", bad
    assert client.post("/api/v1/documents/url", json={}).status_code == 422


# --------------------------------------------------------------------------- SSRF protection (strict default)
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8000/", "http://[::1]/", "http://10.1.2.3/", "http://192.168.0.10/",
    "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/", "http://100.64.0.1/", "http://0.0.0.0/",
])
def test_internal_addresses_are_blocked(url):
    with pytest.raises(WebFetchError) as info:
        validate_public_url(url)
    assert info.value.code == "blocked_host"


def test_strict_api_blocks_loopback_by_default(settings, embeddings):
    app = create_app(settings, resource_factory=lambda s: build_resources(s, embeddings=embeddings, chat_model=None))
    with TestClient(app) as c:
        r = c.post("/api/v1/documents/url", json={"url": "http://127.0.0.1:8000/"})
        assert r.status_code == 422 and r.json()["error"]["code"] == "url_not_allowed"


def test_hostname_resolving_to_a_private_address_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 80))])
    with pytest.raises(WebFetchError) as info:
        validate_public_url("http://innocent-looking.example.com/")
    assert info.value.code == "blocked_host"


def test_redirect_to_an_internal_address_is_blocked_on_every_hop(monkeypatch):
    """A public-looking first hop that redirects to the cloud metadata address must not be followed."""
    real = socket.getaddrinfo

    def fake_resolver(host, *a, **k):
        if host == "public.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        return real(host, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake_resolver)
    sent = []

    class Redirecting(requests.adapters.BaseAdapter):  # transport only; redirect handling is requests' own
        def send(self, request, **kwargs):
            sent.append(request.url)
            r = requests.Response()
            r.status_code, r.url, r.request = 302, request.url, request
            r.headers["Location"] = "http://169.254.169.254/latest/meta-data/"
            r.raw = __import__("io").BytesIO(b"")
            return r

        def close(self):
            pass

    session = SafeSession(max_bytes=10_000, max_redirects=5)
    session.mount("http://", Redirecting())
    with pytest.raises(WebFetchError) as info:
        session.get("http://public.example/start", timeout=2)
    assert info.value.code == "blocked_host" and sent == ["http://public.example/start"]  # the internal hop was never sent


def test_loader_raises_typed_errors_instead_of_returning_nothing(web):
    loader = WebLoader(allow_private=True, timeout=1, max_bytes=100_000)
    assert loader.load(f"{web}/docs/retry")[0].metadata["source_url"] == f"{web}/docs/retry"
    with pytest.raises(WebFetchError) as info:
        loader.load(f"{web}/missing")
    assert info.value.code == "http_status"
