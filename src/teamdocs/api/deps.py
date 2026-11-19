from __future__ import annotations

from fastapi import Request

from teamdocs.services.container import AppResources
from teamdocs.services.errors import TeamDocsError


class ServiceUnavailableError(TeamDocsError):
    code, status_code = "service_unavailable", 503


def get_resources(request: Request) -> AppResources:
    resources: AppResources | None = getattr(request.app.state, "resources", None)
    if resources is None:
        raise ServiceUnavailableError(
            "The service is not ready: required resources failed to initialise. See /ready."
        )
    return resources
