"""Per-agent model policy: OpenRouter when configured, local Ollama otherwise.

One :class:`ModelManager` is created per application. It is the *only* place that decides which
provider/model an agent (``main``, ``rag``, ``github``) uses, builds and caches the chat models,
runs the single controlled fallback, and reports the real provider status.

Policy (``LLM_PROVIDER`` and the per-agent ``MAIN_/RAG_/GITHUB_LLM_PROVIDER`` overrides):

* ``auto``       - OpenRouter if a key is configured, otherwise Ollama.
* ``openrouter`` - OpenRouter only; a missing key is an explicit error (no silent substitution).
* ``ollama``     - local Ollama only.

*Missing key* (a configuration state) is never confused with *rejected key* or *provider outage*
(runtime conditions). In ``auto`` mode an outage/rejection triggers at most **one** retry on Ollama
and the result always reports that a fallback happened and why.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import httpx

from rag.llm import LLMErrorKind, build_chat_model, build_ollama_model, classify_llm_error

if TYPE_CHECKING:  # pragma: no cover
    from teamdocs.core.config import Settings

logger = logging.getLogger(__name__)

AgentRole = Literal["main", "rag", "github"]
AGENT_ROLES: tuple[str, ...] = ("main", "rag", "github")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][\w.\-:/@+]{0,199}$")
_FALLBACK_KINDS = {
    LLMErrorKind.AUTH,
    LLMErrorKind.RATE_LIMIT,
    LLMErrorKind.TIMEOUT,
    LLMErrorKind.NETWORK,
    LLMErrorKind.PROVIDER,
}
T = TypeVar("T")


class ModelUnavailableError(RuntimeError):
    """No usable model for an agent. The message is safe to show to users."""


@dataclass(frozen=True, slots=True)
class ModelChoice:
    provider: str  # "openrouter" | "ollama" | "injected"
    model: str
    reason: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(slots=True)
class ModelHandle:
    model: Any
    choice: ModelChoice
    fallback_from: ModelChoice | None = None
    fallback_kind: str | None = None

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "provider": self.choice.provider,
            "model": self.choice.model,
            "reason": self.choice.reason,
            "fallback": False,
        }
        if self.fallback_from is not None:
            info.update(
                fallback=True,
                fallback_from=self.fallback_from.label,
                fallback_reason=self.fallback_kind,
            )
        return info


def _tag(exc: BaseException, choice: ModelChoice) -> None:
    """Remember which model was in use when ``exc`` happened (for precise user-facing errors)."""
    try:
        exc.teamdocs_model_choice = choice  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - exceptions with __slots__
        pass


def _norm(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class ModelManager:
    def __init__(
        self,
        settings: Settings,
        *,
        fixed_model: Any | None = None,
        model_factory: Callable[[ModelChoice, Settings], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._fixed = fixed_model  # injected chat model (tests/benchmarks); bypasses the policy
        self._factory = model_factory  # optional builder replacing the real OpenRouter/Ollama clients (tests)
        self._override: dict[str, str] = {}
        self._cache: dict[tuple[str, str], Any] = {}
        self._lock = threading.RLock()

    @property
    def configured(self) -> bool:
        """An OpenRouter key (or an injected model) exists. Ollama needs no key, so it does not count."""
        return self._fixed is not None or self._settings.openrouter_api_key is not None

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> Settings:
        return self._settings

    def reload(self, settings: Settings) -> None:
        """Adopt freshly loaded settings (new key / model) and drop all cached clients."""
        with self._lock:
            self._settings = settings
            self._cache.clear()

    def set_override(
        self,
        *,
        provider: str | None = None,
        openrouter_model: str | None = None,
        ollama_model: str | None = None,
    ) -> dict[str, str]:
        """Runtime selection from the UI (kept in memory only; ``.env`` is never rewritten)."""
        updates: dict[str, str | None] = {
            "provider": provider,
            "openrouter_model": openrouter_model,
            "ollama_model": ollama_model,
        }
        with self._lock:
            for key, value in updates.items():
                if value is None:
                    continue
                value = value.strip()
                if not value or value == "default":
                    self._override.pop(key, None)
                    continue
                if key == "provider" and value not in ("auto", "openrouter", "ollama"):
                    raise ValueError("provider must be one of: auto, openrouter, ollama")
                if key != "provider" and not _MODEL_NAME.match(value):
                    raise ValueError(f"invalid model name for {key}")
                self._override[key] = value
            self._cache.clear()
            return dict(self._override)

    @property
    def override(self) -> dict[str, str]:
        return dict(self._override)

    # -------------------------------------------------------------------- policy
    def _policy(self, role: str) -> str:
        s = self._settings
        return self._override.get("provider") or getattr(s, f"{role}_llm_provider", None) or s.llm_provider

    def _openrouter_model(self) -> str:
        return self._override.get("openrouter_model") or self._settings.openrouter_model

    def _ollama_model(self) -> str:
        return self._override.get("ollama_model") or self._settings.ollama_model

    def choose(self, role: str) -> ModelChoice:
        if self._fixed is not None:
            return ModelChoice("injected", type(self._fixed).__name__, "model injected by the caller")
        policy = self._policy(role)
        has_key = self._settings.openrouter_api_key is not None
        if policy == "openrouter":
            if not has_key:
                raise ModelUnavailableError(
                    f"The {role} agent is set to OpenRouter but OPENROUTER_API_KEY is not configured. "
                    "Set the key in .env (then reload settings) or choose provider 'auto'/'ollama'."
                )
            return ModelChoice("openrouter", self._openrouter_model(), f"{role}: OpenRouter selected explicitly")
        if policy == "ollama":
            return ModelChoice("ollama", self._ollama_model(), f"{role}: Ollama selected explicitly")
        if has_key:
            return ModelChoice("openrouter", self._openrouter_model(), f"{role}: OPENROUTER_API_KEY is configured")
        return ModelChoice(
            "ollama", self._ollama_model(), f"{role}: no OpenRouter key configured, using local Ollama"
        )

    # ------------------------------------------------------------------- models
    def _build(self, choice: ModelChoice) -> Any:
        s = self._settings
        key = (choice.provider, choice.model)
        with self._lock:
            if key not in self._cache:
                if self._factory is not None:
                    self._cache[key] = self._factory(choice, s)
                elif choice.provider == "openrouter":
                    self._cache[key] = build_chat_model(
                        s.openrouter_api_key.get_secret_value(),  # type: ignore[union-attr]
                        choice.model,
                        timeout_seconds=s.llm_timeout_seconds,
                        max_tokens=s.llm_max_tokens,
                        max_retries=s.llm_max_retries,
                        base_url=s.openrouter_base_url,
                    )
                else:
                    self._cache[key] = build_ollama_model(
                        choice.model,
                        base_url=s.ollama_base_url,
                        timeout_seconds=max(s.llm_timeout_seconds, 120.0),
                        max_tokens=s.llm_max_tokens,
                        num_ctx=s.ollama_num_ctx,
                    )
            return self._cache[key]

    def get(self, role: str) -> ModelHandle:
        choice = self.choose(role)
        if self._fixed is not None:
            return ModelHandle(self._fixed, choice)
        return ModelHandle(self._build(choice), choice)

    def fallback(self, role: str, failed: ModelChoice, kind: LLMErrorKind) -> ModelHandle | None:
        """The single controlled fallback: OpenRouter -> Ollama, only when allowed by the policy."""
        s = self._settings
        if self._fixed is not None or failed.provider != "openrouter":
            return None
        if self._policy(role) != "auto" or not s.ollama_fallback_on_error or kind not in _FALLBACK_KINDS:
            return None
        choice = ModelChoice("ollama", self._ollama_model(), f"{role}: fallback after OpenRouter {kind.value}")
        return ModelHandle(self._build(choice), choice, fallback_from=failed, fallback_kind=kind.value)

    def run(self, role: str, fn: Callable[[ModelHandle], T]) -> tuple[T, ModelHandle]:
        """Run ``fn(handle)``; on a provider failure retry **once** on the fallback model."""
        handle = self.get(role)
        try:
            return fn(handle), handle
        except Exception as exc:
            kind = classify_llm_error(exc)
            fallback = self.fallback(role, handle.choice, kind)
            if fallback is None:
                _tag(exc, handle.choice)
                raise
            logger.warning(
                "%s agent: %s failed (%s); one retry on %s", role, handle.choice.label, kind.value, fallback.choice.label
            )
            try:
                return fn(fallback), fallback
            except Exception as exc2:
                _tag(exc2, fallback.choice)
                raise

    async def arun(self, role: str, fn: Callable[[ModelHandle], Any]) -> tuple[Any, ModelHandle]:
        """Async twin of :meth:`run` (``fn`` is a coroutine function)."""
        handle = self.get(role)
        try:
            return await fn(handle), handle
        except Exception as exc:
            kind = classify_llm_error(exc)
            fallback = self.fallback(role, handle.choice, kind)
            if fallback is None:
                _tag(exc, handle.choice)
                raise
            logger.warning(
                "%s agent: %s failed (%s); one retry on %s", role, handle.choice.label, kind.value, fallback.choice.label
            )
            try:
                return await fn(fallback), fallback
            except Exception as exc2:
                _tag(exc2, fallback.choice)
                raise

    # -------------------------------------------------------------------- status
    def status(self, *, check: bool = False) -> dict[str, Any]:
        """Real provider state. ``check=True`` performs live network probes (no secrets returned)."""
        s = self._settings
        openrouter: dict[str, Any] = {
            "configured": s.openrouter_api_key is not None,
            "key_source": s.key_source("openrouter_api_key"),
            "model": self._openrouter_model(),
            "state": "not_configured" if s.openrouter_api_key is None else "not_checked",
            "detail": "",
        }
        ollama: dict[str, Any] = {
            "base_url": s.ollama_base_url,
            "model": self._ollama_model(),
            "state": "not_checked",
            "installed_models": [],
            "detail": "",
        }
        if check:
            if s.openrouter_api_key is not None:
                openrouter.update(self.check_openrouter())
            ollama.update(self.check_ollama())
        agents: dict[str, Any] = {}
        for role in AGENT_ROLES:
            try:
                choice = self.choose(role)
                agents[role] = {"provider": choice.provider, "model": choice.model, "reason": choice.reason}
            except ModelUnavailableError as exc:
                agents[role] = {"provider": None, "model": None, "error": str(exc)}
        return {
            "policy": self._override.get("provider") or s.llm_provider,
            "override": self.override,
            "env_files_found": s.env_files_found,
            "openrouter": openrouter,
            "ollama": ollama,
            "agents": agents,
            "checked": check,
        }

    def check_openrouter(self, timeout: float = 8.0) -> dict[str, str]:
        """Validate the configured key with a real request. Distinguishes bad key from no network."""
        s = self._settings
        if s.openrouter_api_key is None:
            return {"state": "not_configured", "detail": "OPENROUTER_API_KEY is not set."}
        headers = {"Authorization": f"Bearer {s.openrouter_api_key.get_secret_value()}"}
        try:
            for path in ("/key", "/auth/key"):
                response = httpx.get(f"{s.openrouter_base_url}{path}", headers=headers, timeout=timeout)
                if response.status_code != 404:
                    break
        except httpx.TimeoutException:
            return {"state": "unreachable", "detail": "OpenRouter did not answer in time."}
        except httpx.HTTPError as exc:
            return {"state": "unreachable", "detail": f"Cannot reach OpenRouter ({type(exc).__name__})."}
        if response.status_code in (401, 403):
            return {"state": "invalid_key", "detail": "OpenRouter rejected the configured key."}
        if response.status_code == 200:
            return {"state": "ok", "detail": "Key accepted by OpenRouter."}
        if response.status_code == 429:
            return {"state": "rate_limited", "detail": "OpenRouter is rate limiting this key."}
        return {"state": "error", "detail": f"OpenRouter answered HTTP {response.status_code}."}

    def check_ollama(self, timeout: float = 5.0) -> dict[str, Any]:
        """List the installed Ollama models and check that the configured one is present."""
        s = self._settings
        try:
            response = httpx.get(f"{s.ollama_base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", []) if isinstance(m, dict)]
        except httpx.HTTPError as exc:
            return {
                "state": "unreachable",
                "installed_models": [],
                "detail": f"Ollama is not reachable at {s.ollama_base_url} ({type(exc).__name__}). Start it with 'ollama serve'.",
            }
        except ValueError:
            return {"state": "error", "installed_models": [], "detail": "Unexpected answer from Ollama."}
        wanted = self._ollama_model()
        if _norm(wanted) in {_norm(n) for n in names}:
            return {"state": "ok", "installed_models": names, "detail": f"Model '{wanted}' is installed."}
        return {
            "state": "model_missing",
            "installed_models": names,
            "detail": f"Model '{wanted}' is not installed. Run: ollama pull {wanted}",
        }


# Backwards-compatible helper for code that still builds a model without a manager.
def get_chat_model(api_key: str | None = None, *, settings: Settings | None = None):
    from teamdocs.core.config import get_settings

    settings = settings or get_settings()
    key = api_key or (settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else "")
    return build_chat_model(
        key,
        settings.openrouter_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_tokens=2048,
        max_retries=settings.llm_max_retries,
        base_url=settings.openrouter_base_url,
    )
