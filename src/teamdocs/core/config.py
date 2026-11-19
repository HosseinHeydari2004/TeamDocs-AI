"""Central application configuration (environment variables / .env file)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# <repo>/src/teamdocs/core/config.py -> <repo>. Used so that ``.env`` is found no matter which
# directory the server is started from (the working directory is checked as well).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILES: tuple[Path, ...] = (PROJECT_ROOT / ".env", Path(".env"))

Provider = Literal["openrouter", "ollama"]
ProviderPolicy = Literal["auto", "openrouter", "ollama"]

_PLACEHOLDER_PREFIXES = ("your_", "your-", "changeme", "sk-or-xxx", "ghp_xxx", "<", "replace")


def clean_secret(value: object) -> object:
    """Normalise a secret read from the environment / ``.env``.

    Handles the usual copy-paste accidents: surrounding whitespace or quotes, a ``Bearer `` prefix,
    a UTF-8 BOM, and template placeholders. Returns ``None`` when nothing usable is left.
    """
    if not isinstance(value, str):
        return value
    value = value.replace("\ufeff", "").strip().strip("\"'").strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if not value or value.lower().startswith(_PLACEHOLDER_PREFIXES):
        return None
    return value


class _NonBlankEnvSource(EnvSettingsSource):
    """Process environment source that ignores *blank* variables.

    Without this, an empty ``OPENROUTER_API_KEY=`` that merely exists in the shell/IDE/Docker
    environment silently shadows the real key written in ``.env``.
    """

    def __call__(self) -> dict[str, Any]:
        data = super().__call__()
        return {k: v for k, v in data.items() if not (isinstance(v, str) and not v.strip())}

ALLOWED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".txt",
        ".md",
        ".docx",
        ".pptx",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".kt",
        ".sql",
        ".html",
        ".htm",
    }
)


class Settings(BaseSettings):
    """All runtime settings. Every field can be overridden by an environment variable
    with the same (case-insensitive) name, or by a ``.env`` file in the working directory."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8-sig",  # tolerate a BOM written by Windows editors
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit arguments > non-blank process environment > .env files.
        return (
            init_settings,
            _NonBlankEnvSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    # --- LLM providers ------------------------------------------------------
    # LLM_PROVIDER: auto = OpenRouter when a key is configured, otherwise local Ollama.
    # The *_LLM_PROVIDER variables override the policy for one agent only.
    llm_provider: ProviderPolicy = "auto"
    main_llm_provider: ProviderPolicy | None = None
    rag_llm_provider: ProviderPolicy | None = None
    github_llm_provider: ProviderPolicy | None = None

    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "openrouter/free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_num_ctx: int = Field(8192, ge=2048, le=131072)  # the Deep Agents prompt + tools need more than Ollama's small default
    ollama_fallback_on_error: bool = True  # one controlled retry on Ollama when OpenRouter is down/rejects the key

    llm_timeout_seconds: float = Field(60.0, gt=0, le=600)
    llm_max_tokens: int = Field(1024, ge=64, le=8192)
    llm_max_retries: int = Field(2, ge=0, le=6)

    # --- GitHub (independent of the LLM provider) -----------------------------
    github_personal_access_token: SecretStr | None = Field(
        None, validation_alias=AliasChoices("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN")
    )
    github_mcp_command: str = ""  # empty = run the official server in Docker; or a path to a local github-mcp-server binary
    github_mcp_image: str = "ghcr.io/github/github-mcp-server"
    github_timeout_seconds: float = Field(60.0, gt=0, le=600)
    github_index_max_files: int = Field(200, ge=1, le=2000)

    # --- Agents (limits that keep runs bounded) -------------------------------
    agent_recursion_limit: int = Field(30, ge=5, le=200)
    main_agent_max_tool_calls: int = Field(6, ge=1, le=30)
    sub_agent_max_tool_calls: int = Field(5, ge=1, le=30)
    agent_timeout_seconds: float = Field(180.0, gt=5, le=1800)

    # --- Web loader (URL ingestion) -------------------------------------------
    web_fetch_timeout_seconds: float = Field(15.0, gt=0, le=120)
    web_max_bytes: int = Field(5 * 1024 * 1024, ge=10_000, le=50 * 1024 * 1024)
    web_max_redirects: int = Field(5, ge=0, le=10)
    web_allow_private_hosts: bool = False  # keep False in production (SSRF protection)

    # --- Embeddings ---------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str | None = None  # None = library default (cpu/cuda auto)
    model_cache_dir: Path = Path("data/models")

    # --- Storage ------------------------------------------------------------
    data_dir: Path = Path("data")
    collection_name: str = Field("teamdocs-ai", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")

    # --- Retrieval ----------------------------------------------------------
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    fetch_k: int = Field(20, ge=1, le=100)
    rrf_k: int = Field(30, ge=1, le=1000)
    dense_weight: float = Field(0.7, ge=0.0, le=1.0)
    min_dense_similarity: float | None = Field(None, ge=-1.0, le=1.0)
    max_context_chars: int = Field(12_000, ge=500, le=100_000)

    # --- Ingestion ----------------------------------------------------------
    chunk_size: int = Field(1200, ge=200, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=2000)
    max_upload_mb: int = Field(25, ge=1, le=500)
    metadata_extraction: bool = False  # LLM-based semantic metadata (needs OPENROUTER_API_KEY)
    pdf_backend: Literal["pypdf", "unstructured"] = "pypdf"

    # --- Server -------------------------------------------------------------
    app_host: str = "127.0.0.1"
    app_port: int = Field(8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_origins: str = ""  # comma separated; empty = same-origin only
    frontend_api_base_url: str = ""  # empty = the frontend talks to the server that served it
    frontend_dir: Path | None = None  # default: <repo>/frontend

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("openrouter_api_key", "github_personal_access_token", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        return clean_secret(value)

    @field_validator("main_llm_provider", "rag_llm_provider", "github_llm_provider", mode="before")
    @classmethod
    def _blank_provider_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _default_provider(cls, value: object) -> object:
        return "auto" if isinstance(value, str) and not value.strip() else value

    @field_validator("openrouter_base_url", "ollama_base_url", mode="before")
    @classmethod
    def _strip_url(cls, value: object) -> object:
        return value.strip().rstrip("/") if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.metadata_extraction and self.openrouter_api_key is None:
            raise ValueError("METADATA_EXTRACTION=true requires OPENROUTER_API_KEY")
        return self

    # --- derived ------------------------------------------------------------
    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_configured(self) -> bool:
        """True when an OpenRouter key is configured (says nothing about whether it is valid)."""
        return self.openrouter_api_key is not None

    @property
    def github_configured(self) -> bool:
        return self.github_personal_access_token is not None

    @property
    def env_files_found(self) -> list[str]:
        """Names of the ``.env`` files that exist and were read (paths only, never contents)."""
        found: list[str] = []
        for path in ENV_FILES:
            resolved = path.resolve()
            if resolved.is_file() and str(resolved) not in found:
                found.append(str(resolved))
        return found

    def key_source(self, name: str) -> str:
        """Where a secret came from, for diagnostics: ``environment`` / ``.env`` / ``not set``."""
        field = self.model_fields[name]
        aliases = field.validation_alias.choices if field.validation_alias else (name,)
        for alias in aliases:
            if clean_secret(os.environ.get(str(alias).upper(), "")):
                return "environment"
        value = getattr(self, name)
        return ".env file" if value is not None else "not set"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.upload_dir, self.tmp_dir, self.model_cache_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Re-read the environment and ``.env`` files (used by ``POST /api/v1/settings/reload``)."""
    get_settings.cache_clear()
    return get_settings()
