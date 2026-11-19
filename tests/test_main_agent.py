"""Main Agent (Deep Agents) + RAG / GitHub sub-agents.

What is real here: the Deep Agents graph, LangChain agents, tools, middleware limits, the vector store,
retrieval, citation handling and the HTTP layer. What is replaced: the *language model* (scripted, because no
API key / Ollama is available in the test environment) and the GitHub MCP session (fake). Therefore these tests
prove wiring, routing mechanics, limits and error handling - not the routing *quality* of a real LLM.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from teamdocs.api.app import create_app
from teamdocs.services.container import build_resources

from .helpers import (
    FakeResult,
    ScriptedToolModel,
    fake_github_factory,
    supervisor_policy,
    tool_call,
)

DOC = b"# Retries\n\nThe client retries failed requests using exponential backoff with jitter.\n"


def make_client(settings, embeddings, policy=None, github=None, **setting_overrides):
    model = ScriptedToolModel(policy=policy or supervisor_policy(), calls=[], seen_tools=[])
    cfg = settings.model_copy(update=setting_overrides) if setting_overrides else settings
    app = create_app(
        cfg,
        resource_factory=lambda s: build_resources(s, embeddings=embeddings, chat_model=model, github_factory=github),
    )
    client = TestClient(app)
    client.__enter__()
    client.model = model
    client.post("/api/v1/documents", files={"file": ("retry.md", io.BytesIO(DOC), "text/markdown")})
    return client


@pytest.fixture
def gh_ok():
    return fake_github_factory(lambda name, args: FakeResult("# Hello-World\nMy first repository on GitHub!"))


@pytest.fixture
def make(settings, embeddings):
    clients = []

    def _make(**kw):
        c = make_client(settings, embeddings, **kw)
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def chat(client, message, history=None):
    return client.post("/api/v1/chat", json={"message": message, "history": history or []})


# --------------------------------------------------------------------------- Deep Agents wiring
def test_main_agent_only_offers_the_two_supervisor_tools(make, gh_ok):
    client = make(github=gh_ok)
    assert chat(client, "hello").status_code == 200
    assert client.model.seen_tools[0] == ["ask_rag_agent", "ask_github_agent"]  # no fs/shell/todo/delete tools


# --------------------------------------------------------------------------- routing
def test_routes_to_rag_only_and_returns_structured_sources(make, gh_ok):
    client = make(github=gh_ok)
    body = chat(client, "What does the document say about retry?").json()
    assert [a["agent"] for a in body["agents_used"]] == ["rag"] and body["agents_used"][0]["ok"]
    assert body["sources"][0]["filename"] == "retry.md" and body["sources"][0]["cited"] is True
    assert "[1]" in body["answer"] and body["github"] == [] and body["warnings"] == []
    assert gh_ok.client.calls == []  # the GitHub sub-agent was never touched


def test_routes_to_github_only(make, gh_ok):
    client = make(github=gh_ok)
    body = chat(client, "Summarise the repository").json()
    assert [a["agent"] for a in body["agents_used"]] == ["github"] and body["agents_used"][0]["ok"]
    assert body["sources"] == [] and body["github"][0]["label"] == "octocat/Hello-World:README.md"
    assert gh_ok.client.calls[0][0] == "get_file_contents"
    assert gh_ok.client.calls[0][1] == {"owner": "octocat", "repo": "Hello-World", "path": "README.md"}


def test_combined_request_calls_both_agents_and_merges(make, gh_ok):
    client = make(github=gh_ok)
    body = chat(client, "Compare the retry document with the repository").json()
    assert sorted(a["agent"] for a in body["agents_used"]) == ["github", "rag"]
    assert body["sources"] and body["github"]
    assert "backoff" in body["answer"] and "Hello-World" in body["answer"]


def test_ambiguous_request_asks_for_clarification_without_calling_agents(make, gh_ok):
    client = make(github=gh_ok)
    body = chat(client, "What is in that repo?").json()
    assert body["agents_used"] == [] and body["answer"].endswith("?")
    assert gh_ok.client.calls == []


def test_history_is_passed_to_the_main_agent(make, gh_ok):
    client = make(github=gh_ok)
    chat(client, "hello again", history=[{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}])
    texts = [str(m.content) for m in client.model.calls[0]]
    assert any("earlier question" in t for t in texts) and any("earlier answer" in t for t in texts)


# --------------------------------------------------------------------------- failures are reported, never hidden
def test_github_not_configured_does_not_block_rag(make):
    client = make()  # default GitHub factory: no GITHUB_PERSONAL_ACCESS_TOKEN
    body = chat(client, "Compare the retry document with the repository").json()
    by_agent = {a["agent"]: a for a in body["agents_used"]}
    assert by_agent["rag"]["ok"] is True
    assert by_agent["github"]["ok"] is False and by_agent["github"]["error"]["code"] == "github_not_configured"
    assert any("github agent failed" in w.lower() for w in body["warnings"])
    assert body["sources"]  # the RAG half still delivered evidence


def test_github_bad_token_is_reported_as_failure_not_success(make):
    bad = fake_github_factory(lambda n, a: FakeResult("failed to get repository info: 401 Bad credentials", is_error=True))
    client = make(github=bad)
    body = chat(client, "Summarise the repository").json()
    agent = body["agents_used"][0]
    assert agent["ok"] is False and agent["error"]["code"] == "github_auth"
    assert body["github"] == [] and body["warnings"]


def test_github_tool_exception_is_contained(make):
    def boom(name, args):
        raise RuntimeError("mcp server crashed")

    client = make(github=fake_github_factory(boom))
    agent = chat(client, "Summarise the repository").json()["agents_used"][0]
    assert agent["ok"] is False and agent["error"]["code"] == "github_error"


def test_rag_answer_without_retrieval_is_never_passed_off_as_knowledge(make, gh_ok):
    client = make(github=gh_ok, policy=supervisor_policy())
    body = chat(client, "Check the document, no tool please").json()  # sub-agent model answers w/o searching
    rag = body["agents_used"][0]
    assert rag["ok"] and rag["insufficient_evidence"] is True
    assert "retries are great" not in body["answer"] and body["sources"] == []


def test_fabricated_citations_are_stripped(make, gh_ok):
    client = make(github=gh_ok, policy=supervisor_policy(rag_final="Backoff is used [1] and also [7]."))
    body = chat(client, "What does the document say about retry?").json()
    assert "[1]" in body["answer"] and "[7]" not in body["answer"]


# --------------------------------------------------------------------------- limits / loops
def test_duplicate_subagent_calls_are_suppressed(make, gh_ok):
    state = {"n": 0}
    inner = supervisor_policy()

    def policy(messages, tools):
        if "ask_rag_agent" in tools and state["n"] < 2 and messages[-1].type != "tool":
            state["n"] += 1
            return AIMessage(content="", tool_calls=[tool_call("ask_rag_agent", {"question": "How does the retry policy work?"}, f"c{state['n']}")])
        return inner(messages, tools)

    client = make(github=gh_ok, policy=policy)
    body = chat(client, "What does the document say about retry?").json()
    assert len([a for a in body["agents_used"] if a["agent"] == "rag"]) == 1


def test_runaway_tool_loop_is_bounded(make, gh_ok):
    counter = {"n": 0}

    def looping(messages, tools):
        if "ask_rag_agent" in tools:
            counter["n"] += 1
            return AIMessage(content="", tool_calls=[tool_call("ask_rag_agent", {"question": f"question {counter['n']}"}, f"id{counter['n']}")])
        return supervisor_policy()(messages, tools)

    client = make(github=gh_ok, policy=looping, main_agent_max_tool_calls=2)
    response = chat(client, "What does the document say about retry?")
    assert counter["n"] <= 2 + 3  # tool limit 2 + model-call backstop; no infinite loop
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        assert len(response.json()["agents_used"]) <= 2
    else:
        assert response.json()["error"]["code"] == "agent_limit"


# --------------------------------------------------------------------------- API contract
def test_chat_validation_errors(make, gh_ok):
    client = make(github=gh_ok)
    for payload in ({}, {"message": ""}, {"message": "   "}, {"message": "x" * 4001}, {"message": "hi", "history": [{"role": "system", "content": "x"}]}):
        r = client.post("/api/v1/chat", json=payload)
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_no_secret_in_any_status_or_chat_payload(settings, embeddings):
    secret = "sk-or-v1-TOPSECRETVALUE"
    gh_secret = "ghp_TOPSECRETGITHUB"
    cfg = settings.model_copy(update={"openrouter_api_key": secret, "github_personal_access_token": gh_secret})
    client = make_client(cfg, embeddings)
    try:
        blobs = [client.get("/api/v1/status").text, client.get("/ready").text, chat(client, "hello").text, client.post("/api/v1/settings/reload").text]
        assert all(secret not in b and gh_secret not in b for b in blobs)
    finally:
        client.__exit__(None, None, None)
