"""Turn any failure inside an agent run into a stable ``code`` + user-safe message."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from devmind.github.errors import GitHubAuthError, GitHubNotConfiguredError, GitHubUnavailableError
from devmind.llm import ModelUnavailableError
from rag.llm import LLMErrorKind, classify_llm_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentFailure:
    code: str
    message: str


def describe_failure(exc: BaseException) -> AgentFailure:
    if isinstance(exc, ModelUnavailableError):
        return AgentFailure("llm_not_configured", str(exc))
    if isinstance(exc, GitHubNotConfiguredError):
        return AgentFailure("github_not_configured", str(exc))
    if isinstance(exc, GitHubAuthError):
        return AgentFailure("github_auth", str(exc))
    if isinstance(exc, GitHubUnavailableError):
        return AgentFailure("github_unavailable", str(exc))

    if type(exc).__name__ == "GraphRecursionError":
        return AgentFailure("agent_limit", "The agent reached its step limit without finishing. Try a more specific question.")

    choice = getattr(exc, "teamdocs_model_choice", None)
    provider = getattr(choice, "provider", None) or "the language model provider"
    name = "Ollama" if provider == "ollama" else "OpenRouter" if provider == "openrouter" else provider
    text = str(exc).lower()
    kind = classify_llm_error(exc)

    if "does not support tools" in text or "does not support function" in text:
        return AgentFailure(
            "llm_no_tool_support",
            f"The selected {name} model does not support tool calling, which the agents need. "
            "Choose a tool-capable model (for Ollama e.g. qwen2.5 or llama3.1).",
        )
    if kind is LLMErrorKind.AUTH:
        return AgentFailure("llm_auth", f"{name} rejected the configured credentials. Check the API key.")
    if kind is LLMErrorKind.RATE_LIMIT:
        return AgentFailure("llm_rate_limited", f"{name} is rate limiting requests. Retry shortly.")
    if kind is LLMErrorKind.TIMEOUT:
        return AgentFailure("llm_timeout", f"{name} took too long to respond. Please try again.")
    if kind is LLMErrorKind.MODEL_NOT_FOUND:
        hint = " Run: ollama pull <model>." if provider == "ollama" else " Check the model name."
        return AgentFailure("llm_model_not_found", f"The selected {name} model was not found.{hint}")
    if kind is LLMErrorKind.NETWORK:
        hint = " Start it with 'ollama serve' and check OLLAMA_BASE_URL." if provider == "ollama" else " Check your network."
        return AgentFailure("llm_unreachable", f"{name} is not reachable.{hint}")
    if kind is LLMErrorKind.PROVIDER:
        return AgentFailure("llm_provider_error", f"{name} returned a server error. Please try again.")
    logger.error("Unclassified agent failure: %s: %s", type(exc).__name__, str(exc)[:300])
    return AgentFailure("agent_error", "The agent failed unexpectedly. Check the server logs.")
