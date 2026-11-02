from langchain_openrouter import ChatOpenRouter

from devmind.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


def get_chat_model() -> ChatOpenRouter:
    """Create the configured OpenRouter chat model."""

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing from .env."
        )

    return ChatOpenRouter(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        temperature=0,
        max_tokens=512,
        max_retries=2,
    )