from langchain_openrouter import ChatOpenRouter

from devmind.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


def get_chat_model(api_key: str | None = None) -> ChatOpenRouter:
    if api_key is None:
        api_key = OPENROUTER_API_KEY

    api_key = (api_key or "").strip()

    if not api_key:
        raise RuntimeError("OpenRouter API key is required.")

    return ChatOpenRouter(
        model=OPENROUTER_MODEL,
        api_key=api_key,
        temperature=0,
        max_tokens=2048,
        max_retries=2,
    )