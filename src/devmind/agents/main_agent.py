"""Main (supervisor) agent, built with Deep Agents' ``create_deep_agent``.

The user talks to this single agent. It analyses the request with the language model and calls the
RAG and/or GitHub sub-agents through two typed tools; it asks a clarifying question when the request
is ambiguous. Deep Agents' built-in file-system / shell / planning tools and the default
general-purpose sub-agent are switched off on purpose: this agent is exposed through a web API and only
needs the two supervisor tools.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from devmind.agents.common import final_answer, limits
from devmind.agents.contracts import SubAgentResult
from devmind.agents.errors import describe_failure
from devmind.agents.github_agent import GitHubSubAgent
from devmind.agents.prompts import MAIN_AGENT_PROMPT
from devmind.agents.rag_agent import RagSubAgent, SourceRegistry
from devmind.llm import ModelManager

logger = logging.getLogger(__name__)

_DEEP_AGENT_BUILTINS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "write_todos"}
)
_HISTORY_TURNS = 10
_HISTORY_CHARS = 4000


class MainAgentError(RuntimeError):
    """The Main Agent itself failed (not a sub-agent). Carries a stable code and a safe message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ChatRun:
    """State of one chat turn: sub-agent results, shared citation numbers, duplicate-call guard."""

    results: list[SubAgentResult] = field(default_factory=list)
    registry: SourceRegistry = field(default_factory=SourceRegistry)
    seen: set[tuple] = field(default_factory=set)

    def reset(self) -> None:
        self.results.clear()
        self.registry = SourceRegistry()
        self.seen.clear()


@dataclass
class ChatOutcome:
    answer: str
    results: list[SubAgentResult]
    model: dict[str, Any]
    duration_ms: float
    warnings: list[str] = field(default_factory=list)


def restrict_builtin_tools(model: Any) -> None:
    """Register (idempotently) the Deep Agents harness profile that hides the built-in tools."""
    try:
        provider = model._get_ls_params().get("ls_provider")
    except Exception:  # pragma: no cover - exotic model classes
        provider = None
    if provider:
        register_harness_profile(
            provider,
            HarnessProfile(
                excluded_tools=_DEEP_AGENT_BUILTINS,
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            ),
        )


def build_supervisor_tools(run: ChatRun, rag_agent: RagSubAgent, github_agent: GitHubSubAgent) -> list[BaseTool]:
    """The two sub-agents exposed as typed tools. Inputs and outputs are structured."""

    async def delegate(key: tuple, coro_factory) -> str:
        if key in run.seen:
            return json.dumps({"ok": False, "note": "Duplicate call ignored: this exact question was already asked. Use the earlier result."})
        run.seen.add(key)
        result: SubAgentResult = await coro_factory()
        run.results.append(result)
        return json.dumps(result.for_supervisor(), ensure_ascii=False)

    @tool
    async def ask_rag_agent(question: str, document: str | None = None) -> str:
        """Ask the RAG sub-agent. It searches the user's indexed documents and imported web pages and answers with
        [n] citations. Use for any question about documents, stored knowledge or web content.
        `document` optionally restricts the search to one indexed file name or URL."""
        question = (question or "").strip()
        if not question:
            return json.dumps({"ok": False, "error": "question must not be empty"})
        return await delegate(("rag", question.lower(), document), lambda: rag_agent.run(question, document, run.registry))

    @tool
    async def ask_github_agent(question: str, repository: str | None = None) -> str:
        """Ask the GitHub sub-agent (read-only). It analyses repositories, code files, issues, pull requests and
        commits. `repository` is "owner/repo" when known."""
        question = (question or "").strip()
        if not question:
            return json.dumps({"ok": False, "error": "question must not be empty"})
        return await delegate(("github", question.lower(), repository), lambda: github_agent.run(question, repository))

    return [ask_rag_agent, ask_github_agent]


def build_main_agent(model: Any, tools: list[BaseTool], max_tool_calls: int):
    restrict_builtin_tools(model)
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=MAIN_AGENT_PROMPT,
        middleware=limits(max_tool_calls),
        name="teamdocs-main-agent",
    )


class MainAgentService:
    def __init__(self, models: ModelManager, rag_agent: RagSubAgent, github_agent: GitHubSubAgent) -> None:
        self._models = models
        self._rag = rag_agent
        self._github = github_agent

    async def chat(self, message: str, history: list[dict[str, str]] | None = None) -> ChatOutcome:
        started = time.perf_counter()
        settings = self._models.settings
        run = ChatRun()
        tools = build_supervisor_tools(run, self._rag, self._github)

        messages: list[Any] = []
        for turn in (history or [])[-_HISTORY_TURNS:]:
            text = str(turn.get("content", ""))[:_HISTORY_CHARS]
            if not text.strip():
                continue
            messages.append(AIMessage(content=text) if turn.get("role") == "assistant" else HumanMessage(content=text))
        messages.append(HumanMessage(content=message))

        async def attempt(handle):
            run.reset()  # a fallback re-run must not mix with results of the failed attempt
            agent = build_main_agent(handle.model, tools, settings.main_agent_max_tool_calls)
            return await asyncio.wait_for(
                agent.ainvoke({"messages": messages}, config={"recursion_limit": settings.agent_recursion_limit}),
                settings.agent_timeout_seconds,
            )

        try:
            result, handle = await self._models.arun("main", attempt)
        except Exception as exc:
            failure = describe_failure(exc)
            if isinstance(exc, TimeoutError):
                failure = describe_failure(TimeoutError("timed out"))
            raise MainAgentError(failure.code, failure.message) from exc

        answer = final_answer(result)
        if not answer:
            raise MainAgentError("agent_limit", "The assistant did not produce an answer within its limits. Try a more specific question.")

        warnings: list[str] = []
        info = handle.describe()
        if info.get("fallback"):
            warnings.append(f"The main agent used local Ollama because {info['fallback_from']} failed ({info['fallback_reason']}).")
        for item in run.results:
            if not item.ok and item.error:
                warnings.append(f"The {item.agent} agent failed: {item.error.message}")
            elif item.model and item.model.get("fallback"):
                warnings.append(f"The {item.agent} agent used local Ollama because {item.model['fallback_from']} failed ({item.model['fallback_reason']}).")
        return ChatOutcome(answer, list(run.results), info, round((time.perf_counter() - started) * 1000, 1), warnings)
