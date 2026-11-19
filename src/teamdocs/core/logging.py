"""Logging setup. Secrets and document contents are never logged by this package."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format=_FORMAT, force=True)
    # Third-party libraries can log request URLs/headers at DEBUG/INFO level.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb.telemetry", "posthog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
