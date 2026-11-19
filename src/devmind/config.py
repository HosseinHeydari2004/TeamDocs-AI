"""Backwards-compatible view of the central settings.

Historically this module loaded ``.env`` on its own (``load_dotenv()`` at import time), which could
disagree with the FastAPI app's configuration. All configuration now comes from
:class:`teamdocs.core.config.Settings`; the module-level names below are resolved *lazily* so they
always reflect the current settings.
"""
from __future__ import annotations

from teamdocs.core.config import get_settings


def __getattr__(name: str):
    settings = get_settings()
    if name == "OPENROUTER_API_KEY":
        key = settings.openrouter_api_key
        return key.get_secret_value() if key else None
    if name == "OPENROUTER_MODEL":
        return settings.openrouter_model
    if name == "GITHUB_PERSONAL_ACCESS_TOKEN":
        token = settings.github_personal_access_token
        return token.get_secret_value() if token else None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
