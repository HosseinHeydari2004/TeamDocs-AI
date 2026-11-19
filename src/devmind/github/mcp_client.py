"""Client for the official GitHub MCP server (read-only).

The server runs either in Docker (default) or from a local ``github-mcp-server`` binary
(``GITHUB_MCP_COMMAND``). The token is passed through the child's environment only - never on the
command line, in logs, or in error messages.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from devmind.github.errors import GitHubNotConfiguredError, GitHubUnavailableError
from devmind.github.results import result_is_error, result_text

logger = logging.getLogger(__name__)


class GitHubMCPClient:
    """Connect to the GitHub MCP server over stdio."""

    def __init__(
        self,
        token: str | None = None,
        *,
        command: str | None = None,
        image: str | None = None,
        timeout_seconds: float | None = None,
        extra_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ):
        from teamdocs.core.config import get_settings

        settings = get_settings()
        if token is None and settings.github_personal_access_token is not None:
            token = settings.github_personal_access_token.get_secret_value()
        token = (token or "").strip()
        if not token:
            raise GitHubNotConfiguredError(
                "GitHub is not configured: set GITHUB_PERSONAL_ACCESS_TOKEN in .env (a read-only token is enough)."
            )

        self.timeout = timeout_seconds or settings.github_timeout_seconds
        command = settings.github_mcp_command if command is None else command
        image = image or settings.github_mcp_image
        env = {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token, **(extra_env or {})}

        if command:
            args = ["stdio", "--read-only", *(extra_args or [])]
            self._executable = command
            self.server_params = StdioServerParameters(command=command, args=args, env=env)
        else:
            self._executable = "docker"
            self.server_params = StdioServerParameters(
                command="docker",
                args=[
                    "run", "-i", "--rm",
                    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "-e", "GITHUB_READ_ONLY=1",
                    image,
                ],
                env=env,
            )

    def _check_executable(self) -> None:
        if shutil.which(self._executable) is None and not os.path.isfile(self._executable):
            hint = (
                "Install Docker Desktop and start it, or set GITHUB_MCP_COMMAND to a local github-mcp-server binary."
                if self._executable == "docker"
                else "Check the GITHUB_MCP_COMMAND path."
            )
            raise GitHubUnavailableError(f"The GitHub MCP server cannot be started: '{self._executable}' was not found. {hint}")

    @asynccontextmanager
    async def connect(self):
        self._check_executable()
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    try:
                        await asyncio.wait_for(session.initialize(), self.timeout)
                    except TimeoutError as exc:
                        raise GitHubUnavailableError(
                            "The GitHub MCP server did not start in time (is Docker running / image pulled?)."
                        ) from exc
                    yield session
        except FileNotFoundError as exc:
            raise GitHubUnavailableError(f"The GitHub MCP server cannot be started: {exc.filename or self._executable} not found.") from exc

    async def list_tools(self) -> list[dict]:
        async with self.connect() as session:
            result = await asyncio.wait_for(session.list_tools(), self.timeout)
            return [{"name": tool.name, "description": tool.description or ""} for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict | None = None):
        async with self.connect() as session:
            return await asyncio.wait_for(session.call_tool(name, arguments=arguments or {}), self.timeout)


__all__ = ["GitHubMCPClient", "result_is_error", "result_text"]
