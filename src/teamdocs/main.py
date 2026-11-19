"""ASGI entry point:  uvicorn teamdocs.main:app  (or  python -m teamdocs.main)."""

from __future__ import annotations

import uvicorn

from teamdocs.api.app import create_app
from teamdocs.core.config import get_settings

app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "teamdocs.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
