"""Chat-model factories and provider-error classification, shared by RAG, the API and devmind.

Only *how to build a model* lives here. *Which* provider to use (OpenRouter vs. local Ollama)
is a policy decision made in :mod:`devmind.llm` from the central settings.
"""
from __future__ import annotations

import enum

from langchain_openrouter import ChatOpenRouter


def build_chat_model(
        api_key: str,
        model: str = "openrouter/free",
        *,
        timeout_seconds: float = 60.0,
        max_tokens: int = 1024,
        max_retries: int = 2,
        temperature: float = 0.0,
        base_url: str | None = None,
) -> ChatOpenRouter:
    """Create an OpenRouter chat model. The client is built once and reused."""
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("OpenRouter API key is required.")

    extra = {"base_url": base_url} if base_url else {}
    return ChatOpenRouter(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=int(timeout_seconds * 1000),  # the SDK expects milliseconds
        **extra,
    )


def build_ollama_model(
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_ctx: int = 8192,
):
    """Create a chat model for a local Ollama server (no API key involved)."""
    from langchain_ollama import ChatOllama  # imported lazily: only needed when Ollama is used

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=max_tokens,
        num_ctx=num_ctx,
        client_kwargs={"timeout": timeout_seconds},
    )


class LLMErrorKind(str, enum.Enum):
    AUTH = "auth"                    # key rejected (401/403)
    RATE_LIMIT = "rate_limit"        # 429
    TIMEOUT = "timeout"
    NETWORK = "network"              # DNS failure, connection refused / reset
    PROVIDER = "provider"            # 5xx / provider-side failure
    MODEL_NOT_FOUND = "model_not_found"
    OTHER = "other"


def classify_llm_error(exc: BaseException) -> LLMErrorKind:
    """Map any provider/SDK exception to a coarse, provider-independent category.

    Deliberately separates *authentication* problems from *outages* and from a *missing key*:
    only the latter is a configuration state, the others are runtime conditions.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < 5:
        chain.append(current)
        current = current.__cause__ or current.__context__

    for item in chain:
        name = type(item).__name__.lower()
        text = str(item).lower()
        status = getattr(item, "status_code", None) or getattr(getattr(item, "response", None), "status_code", None)

        if status in (401, 403) or "unauthorized" in name or "authentication" in name or "invalid api key" in text or "user not found" in text:
            return LLMErrorKind.AUTH
        if status == 404 and ("model" in text and "not found" in text):
            return LLMErrorKind.MODEL_NOT_FOUND
        if "not found, try pulling it" in text or ("model" in text and "not found" in text and "ollama" in text):
            return LLMErrorKind.MODEL_NOT_FOUND
        if status == 429 or "ratelimit" in name or "rate limit" in text or "too many requests" in text:
            return LLMErrorKind.RATE_LIMIT
        if isinstance(item, TimeoutError) or "timeout" in name or "timed out" in text:
            return LLMErrorKind.TIMEOUT
        if isinstance(item, (ConnectionError, OSError)) and not isinstance(item, TimeoutError):
            return LLMErrorKind.NETWORK
        if "connecterror" in name or "connection" in name or "failed to connect" in text or "connection refused" in text or "name or service not known" in text or "getaddrinfo" in text:
            return LLMErrorKind.NETWORK
        if isinstance(status, int) and status >= 500:
            return LLMErrorKind.PROVIDER
    return LLMErrorKind.OTHER
