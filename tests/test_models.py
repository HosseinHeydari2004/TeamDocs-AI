"""Model policy (OpenRouter / Ollama), controlled fallback, and real status probes.

Status probes talk to REAL local HTTP servers (started in the test), so URL building, headers, timeouts and
status-code classification are exercised for real. The real openrouter.ai / a real Ollama are not reachable here.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from langchain_core.messages import AIMessage

from devmind.llm import ModelManager, ModelUnavailableError
from teamdocs.core.config import Settings, clean_secret

from .helpers import ScriptedToolModel


def settings_with(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


class _Server:
    """Tiny configurable HTTP server running in a thread."""

    def __init__(self, routes: dict[str, tuple[int, dict]]):
        outer = self
        self.requests: list[tuple[str, str | None]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                outer.requests.append((self.path, self.headers.get("Authorization")))
                status, body = routes.get(self.path, (404, {"error": "not found"}))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # silence
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server_factory():
    servers = []

    def make(routes):
        servers.append(_Server(routes))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


# --------------------------------------------------------------------------- policy
def test_auto_without_key_uses_ollama_and_with_key_uses_openrouter():
    no_key = ModelManager(settings_with())
    assert no_key.choose("rag").provider == "ollama" and "no OpenRouter key" in no_key.choose("rag").reason
    with_key = ModelManager(settings_with(openrouter_api_key="sk-or-v1-abc"))
    assert with_key.choose("rag").provider == "openrouter"


def test_explicit_openrouter_without_key_is_an_error_not_a_silent_fallback():
    m = ModelManager(settings_with(llm_provider="openrouter"))
    with pytest.raises(ModelUnavailableError, match="OPENROUTER_API_KEY"):
        m.choose("rag")


def test_per_agent_policy_is_independent():
    m = ModelManager(settings_with(openrouter_api_key="sk-or-v1-abc", rag_llm_provider="ollama"))
    assert m.choose("rag").provider == "ollama"
    assert m.choose("main").provider == "openrouter" and m.choose("github").provider == "openrouter"


def test_runtime_override_and_validation():
    m = ModelManager(settings_with(openrouter_api_key="sk-or-v1-abc"))
    m.set_override(provider="ollama", ollama_model="llama3.1:8b")
    assert (m.choose("rag").provider, m.choose("rag").model) == ("ollama", "llama3.1:8b")
    m.set_override(provider="default")  # clears
    assert m.choose("rag").provider == "openrouter"
    for bad in ({"provider": "gpt"}, {"ollama_model": "bad name; rm -rf /"}):
        with pytest.raises(ValueError):
            m.set_override(**bad)


def test_reload_adopts_a_newly_configured_key():
    m = ModelManager(settings_with())
    assert m.choose("rag").provider == "ollama"
    m.reload(settings_with(openrouter_api_key="sk-or-v1-new"))
    assert m.choose("rag").provider == "openrouter"


# --------------------------------------------------------------------------- controlled fallback
class _AuthError(Exception):
    status_code = 401


def _manager(openrouter_error, **kw):
    built = []

    def factory(choice, _settings):
        built.append(choice.provider)
        if choice.provider == "openrouter":
            return ScriptedToolModel(policy=lambda m, t: openrouter_error, calls=[], seen_tools=[])
        return ScriptedToolModel(policy=lambda m, t: AIMessage(content="from ollama"), calls=[], seen_tools=[])

    return ModelManager(settings_with(openrouter_api_key="sk-or-v1-abc", **kw), model_factory=factory), built


def _invoke(handle):
    return handle.model.invoke("hi").content


@pytest.mark.parametrize("error", [_AuthError("bad key"), TimeoutError("timed out"), httpx.ConnectError("refused")])
def test_openrouter_failure_triggers_exactly_one_fallback_to_ollama(error):
    m, built = _manager(error)
    result, handle = m.run("rag", _invoke)
    assert result == "from ollama" and handle.describe()["fallback"] is True
    assert handle.describe()["fallback_from"].startswith("openrouter:") and built == ["openrouter", "ollama"]


def test_no_fallback_when_policy_is_explicit_or_disabled_or_error_is_not_a_provider_failure():
    for kw, err in ((dict(llm_provider="openrouter"), _AuthError("x")), (dict(ollama_fallback_on_error=False), _AuthError("x")), ({}, ValueError("a bug in our code"))):
        m, _ = _manager(err, **kw)
        with pytest.raises(type(err)):
            m.run("rag", _invoke)


def test_failing_fallback_does_not_retry_again_and_tags_the_failing_model():
    def factory(choice, _s):
        return ScriptedToolModel(policy=lambda m, t: _AuthError("denied") if choice.provider == "openrouter" else httpx.ConnectError("no ollama"), calls=[], seen_tools=[])

    m = ModelManager(settings_with(openrouter_api_key="sk-or-v1-abc"), model_factory=factory)
    with pytest.raises(httpx.ConnectError) as info:
        m.run("rag", _invoke)
    assert info.value.teamdocs_model_choice.provider == "ollama"


# --------------------------------------------------------------------------- real HTTP status probes
def test_ollama_probe_ok_model_missing_and_unreachable(server_factory):
    srv = server_factory({"/api/tags": (200, {"models": [{"name": "qwen2.5:7b-instruct-q4_K_M"}, {"name": "llama3.1:8b"}]})})
    ok = ModelManager(settings_with(ollama_base_url=srv.url)).check_ollama()
    assert ok["state"] == "ok" and "llama3.1:8b" in ok["installed_models"]
    missing = ModelManager(settings_with(ollama_base_url=srv.url, ollama_model="mistral")).check_ollama()
    assert missing["state"] == "model_missing" and "ollama pull mistral" in missing["detail"]
    down = ModelManager(settings_with(ollama_base_url="http://127.0.0.1:9")).check_ollama(timeout=1)
    assert down["state"] == "unreachable" and "ollama serve" in down["detail"]


def test_openrouter_probe_distinguishes_valid_invalid_and_unreachable(server_factory):
    good = server_factory({"/key": (200, {"data": {}})})
    r = ModelManager(settings_with(openrouter_api_key="sk-or-v1-good", openrouter_base_url=good.url)).check_openrouter()
    assert r["state"] == "ok" and good.requests[0][1] == "Bearer sk-or-v1-good"  # key really sent as a Bearer token
    bad = server_factory({"/key": (401, {"error": "invalid"})})
    assert ModelManager(settings_with(openrouter_api_key="sk-or-v1-bad", openrouter_base_url=bad.url)).check_openrouter()["state"] == "invalid_key"
    legacy = server_factory({"/auth/key": (200, {})})  # older API path is tried when /key is 404
    assert ModelManager(settings_with(openrouter_api_key="sk-or-v1-x", openrouter_base_url=legacy.url)).check_openrouter()["state"] == "ok"
    down = ModelManager(settings_with(openrouter_api_key="sk-or-v1-x", openrouter_base_url="http://127.0.0.1:9")).check_openrouter(timeout=1)
    assert down["state"] == "unreachable"
    assert ModelManager(settings_with()).check_openrouter()["state"] == "not_configured"


def test_status_never_contains_the_key(server_factory):
    srv = server_factory({"/key": (200, {}), "/api/tags": (200, {"models": []})})
    s = settings_with(openrouter_api_key="sk-or-v1-SECRET", openrouter_base_url=srv.url, ollama_base_url=srv.url)
    assert "sk-or-v1-SECRET" not in json.dumps(ModelManager(s).status(check=True))


# --------------------------------------------------------------------------- config: the root-cause scenarios
def test_blank_environment_variable_does_not_shadow_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-or-v1-from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # e.g. an empty variable exported by the shell/IDE
    assert Settings(_env_file=env).llm_configured is True


def test_real_environment_variable_wins_over_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-or-v1-from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-from-env")
    assert Settings(_env_file=env).openrouter_api_key.get_secret_value() == "sk-or-v1-from-env"


def test_dotenv_with_bom_and_quotes_is_read(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b'\xef\xbb\xbfOPENROUTER_API_KEY="sk-or-v1-quoted"\nGITHUB_TOKEN=ghp_abc\n')
    s = Settings(_env_file=env)
    assert s.openrouter_api_key.get_secret_value() == "sk-or-v1-quoted"
    assert s.github_personal_access_token.get_secret_value() == "ghp_abc"  # GITHUB_TOKEN alias


def test_env_file_lookup_does_not_depend_on_the_working_directory():
    from teamdocs.core.config import ENV_FILES, PROJECT_ROOT

    assert ENV_FILES[0] == PROJECT_ROOT / ".env" and ENV_FILES[0].is_absolute()


def test_secret_cleanup_rules():
    assert clean_secret("  'sk-or-v1-x'  ") == "sk-or-v1-x"
    assert clean_secret("Bearer sk-or-v1-x") == "sk-or-v1-x"
    for placeholder in ("", "   ", "your_openrouter_key", "changeme", "sk-or-xxx", "ghp_xxxxxxxx", "<paste here>"):
        assert clean_secret(placeholder) is None


def test_github_key_is_independent_from_openrouter():
    s = settings_with(github_personal_access_token="ghp_abc")
    assert s.github_configured and not s.llm_configured
    assert settings_with(openrouter_api_key="sk-or-v1-x").github_configured is False
