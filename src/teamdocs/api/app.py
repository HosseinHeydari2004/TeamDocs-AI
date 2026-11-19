"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from teamdocs import __version__
from teamdocs.api.errors import register_exception_handlers
from teamdocs.api.routes import chat, documents, health, query, status
from teamdocs.core.config import Settings, get_settings
from teamdocs.core.logging import configure_logging
from teamdocs.services.container import AppResources, build_resources

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma telemetry off unless the operator opts in

logger = logging.getLogger(__name__)
_DEFAULT_FRONTEND = Path(__file__).resolve().parents[3] / "frontend"

ResourceFactory = Callable[[Settings], AppResources]


def create_app(settings: Settings | None = None, resource_factory: ResourceFactory | None = None) -> FastAPI:
    """Build the app. ``resource_factory`` lets tests inject fake embeddings / LLMs."""
    settings = settings or get_settings()
    factory = resource_factory or build_resources

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        app.state.resources = None
        app.state.startup_error = None
        logger.info("Starting TeamDocs AI %s", __version__)
        try:
            # Model loading is blocking/CPU heavy: keep it off the event loop.
            app.state.resources = await asyncio.to_thread(factory, settings)
        except Exception as exc:
            # Stay up so /health and /ready can report the problem; never report ready.
            app.state.startup_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.exception("Startup failed; the service will report not-ready")
        yield
        logger.info("Shutting down")

    app = FastAPI(
        title="TeamDocs AI",
        description="AI-powered engineering knowledge assistant: hybrid RAG with verifiable source citations.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    register_exception_handlers(app)

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path not in ("/health", "/ready"):
            logger.info(
                "%s %s -> %d (%.0f ms) rid=%s",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
                request_id,
            )
        return response

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(chat.router)
    app.include_router(status.router)
    app.include_router(documents.router)

    @app.get("/config.js", include_in_schema=False)
    def frontend_config() -> Response:
        """Runtime frontend configuration, so the backend URL needs no source change."""
        base = settings.frontend_api_base_url.rstrip("/")
        body = f"window.TEAMDOCS_CONFIG = {{ apiBaseUrl: {_js_string(base)} }};\n"
        return PlainTextResponse(
            body, media_type="application/javascript", headers={"Cache-Control": "no-store"}
        )

    frontend_dir = settings.frontend_dir or _DEFAULT_FRONTEND
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        logger.warning("Frontend directory not found: %s (API only)", frontend_dir)

    return app


def _js_string(value: str) -> str:
    import json

    return json.dumps(value).replace("</", "<\\/")
