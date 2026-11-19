from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from devmind.github.mcp_client import GitHubMCPClient


class GitHubSession:
    """One open MCP connection reused for several read-only calls (a single sub-agent run)."""

    def __init__(self, session, allowed: set[str], timeout: float):
        self._session = session
        self._allowed = allowed
        self._timeout = timeout

    async def call(self, name: str, arguments: dict | None = None):
        if name not in self._allowed:
            raise ValueError(f"Tool is not allowed by TeamDocs-AI: {name}")
        return await asyncio.wait_for(
            self._session.call_tool(name, arguments=arguments or {}), self._timeout
        )


class GitHubTools:
    """Expose a restricted set of read-only GitHub MCP tools."""

    ALLOWED_TOOLS = {
        "get_file_contents",
        "get_commit",
        "list_branches",
        "list_commits",
        "list_issues",
        "issue_read",
        "list_pull_requests",
        "pull_request_read",
        "search_code",
        "search_issues",
        "search_pull_requests",
        "search_repositories",
        "get_latest_release",
        "list_releases",
        "list_tags",
        "get_tag",
    }

    def __init__(self, client=None, token: str | None = None):
        self.client = client or GitHubMCPClient(token=token)

    async def list_tools(self) -> list[dict]:
        """Return only the tools allowed by this application."""
        tools = await self.client.list_tools()
        return [tool for tool in tools if tool["name"] in self.ALLOWED_TOOLS]

    async def call_read_tool(self, name: str, arguments: dict | None = None):
        """Call an explicitly allowed tool (opens its own connection)."""
        if name not in self.ALLOWED_TOOLS:
            raise ValueError(f"Tool is not allowed by TeamDocs-AI: {name}")
        return await self.client.call_tool(name, arguments or {})

    async def call_read_tools(self, calls: list[tuple[str, dict | None]]) -> list:
        """Execute multiple read-only tools using one MCP connection."""
        for name, _ in calls:
            if name not in self.ALLOWED_TOOLS:
                raise ValueError(f"Tool is not allowed by TeamDocs-AI: {name}")

        results = []
        async with self.client.connect() as session:
            for name, arguments in calls:
                try:
                    results.append(await session.call_tool(name, arguments=arguments or {}))
                except Exception as exc:
                    results.append(exc)
        return results

    @asynccontextmanager
    async def session(self):
        """Open one connection for a whole agent run (enter and exit in the same task)."""
        async with self.client.connect() as raw:
            yield GitHubSession(raw, self.ALLOWED_TOOLS, getattr(self.client, "timeout", 60.0))
