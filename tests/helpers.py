"""Shared test doubles (no network, no model downloads)."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings, L2-normalised.

    Texts that share words get a high cosine similarity, so dense retrieval is
    meaningful in tests without downloading a real model. Call counters let tests
    assert how often the embedder is used.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.document_calls = 0
        self.documents_embedded = 0
        self.query_calls = 0

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        self.documents_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vec(text)


class ScriptedLLM:
    """Minimal chat-model double: returns scripted replies and records every call."""

    def __init__(
        self, reply="The retry policy uses exponential backoff [1].", error: Exception | None = None
    ):
        self.reply = reply
        self.error = error
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        text = self.reply(messages) if callable(self.reply) else self.reply

        class _Msg:
            content = text

        return _Msg()


def make_pdf(pages: list[str]) -> bytes:
    """Build a tiny valid text PDF (one text line per page) without extra libraries."""
    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


# ---------------------------------------------------------------------------------------------
# Tool-calling chat-model double. It replaces ONLY the language model: the Deep Agents graph, the
# sub-agents, the tools, the vector store and the HTTP layer in the tests are real code.
# ---------------------------------------------------------------------------------------------
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class ScriptedToolModel(BaseChatModel):
    """``policy(messages, tool_names) -> AIMessage`` decides each turn (like an LLM would)."""

    policy: object
    calls: list = []
    seen_tools: list = []
    provider: str = "scripted"

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": self.provider, "ls_model_name": "scripted", "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        names = [getattr(t, "name", None) or t.get("name") or t.get("function", {}).get("name") for t in tools]
        self.seen_tools.append(names)
        bound = self.model_copy()
        object.__setattr__(bound, "_tool_names", names)
        return bound

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        msg = self.policy(messages, getattr(self, "_tool_names", []))
        if isinstance(msg, Exception):
            raise msg
        return ChatResult(generations=[ChatGeneration(message=msg)])


def tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def last_tool_text(messages) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "tool":
            return str(m.content)
    return ""


# ---------------------------------------------------------------------------------------------
# GitHub MCP boundary double: replaces only the MCP *session* (the process that talks to GitHub).
# The GitHub tools, validation, recording and sub-agent above it are real code.
# ---------------------------------------------------------------------------------------------
from contextlib import asynccontextmanager  # noqa: E402


class FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeResult:
    def __init__(self, text: str, is_error: bool = False) -> None:
        self.content = [FakeContent(text)]
        self.is_error = is_error  # mcp 2.x attribute name


class FakeMCPClient:
    timeout = 5.0

    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[tuple[str, dict]] = []

    @asynccontextmanager
    async def connect(self):
        client = self

        class Session:
            async def call_tool(self, name, arguments=None):
                client.calls.append((name, arguments or {}))
                return client.responder(name, arguments or {})

        yield Session()


def fake_github_factory(responder):
    from devmind.github.tools import GitHubTools

    client = FakeMCPClient(responder)
    factory = lambda settings: GitHubTools(client=client)  # noqa: E731
    factory.client = client
    return factory


def supervisor_policy(*, rag_final="The client retries with exponential backoff [1].", github_final=None):
    """Scripted 'LLM' for all three agents; it tells them apart by the tools bound to the model."""

    def policy(messages, tools):
        after_human = []
        for m in reversed(messages):
            if m.type == "human":
                break
            if m.type == "tool":
                after_human.append(str(m.content))
        question = next(str(m.content) for m in reversed(messages) if m.type == "human")
        low = question.lower()

        if "ask_rag_agent" in tools:  # ------------------------------ Main Agent
            if after_human:
                return AIMessage(content="FINAL: " + " || ".join(reversed(after_human)))
            if "that repo" in low:
                return AIMessage(content="Which repository do you mean (owner/repo)?")
            calls = []
            if "document" in low or "retry" in low:
                calls.append(tool_call("ask_rag_agent", {"question": "How does the retry policy work?" + (" no tool" if "no tool" in low else "")}, "rag1"))
            if "repository" in low or "issue" in low:
                calls.append(tool_call("ask_github_agent", {"question": "Summarise the README", "repository": "octocat/Hello-World"}, "gh1"))
            return AIMessage(content="", tool_calls=calls) if calls else AIMessage(content="Hello! Ask me about your documents or a GitHub repository.")
        if "search_knowledge_base" in tools:  # ---------------------- RAG sub-agent
            if after_human:
                return AIMessage(content=rag_final if "[1]" in after_human[0] else "I found nothing relevant.")
            if "no tool" in low:
                return AIMessage(content="From my own knowledge: retries are great [3].")
            return AIMessage(content="", tool_calls=[tool_call("search_knowledge_base", {"query": "retry backoff"})])
        if "github_get_file" in tools:  # ----------------------------- GitHub sub-agent
            if after_human:
                return AIMessage(content=github_final or "README: " + after_human[0][:60])
            return AIMessage(content="", tool_calls=[tool_call("github_get_file", {"owner": "octocat", "repo": "Hello-World", "path": "README.md"})])
        return AIMessage(content=f"unexpected tool set: {tools}")

    return policy
