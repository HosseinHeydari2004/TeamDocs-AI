
import os
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class GitHubMCPClient:
    """Connect to the official GitHub MCP server through Docker."""

    def __init__(self):
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

        if not token:
            raise RuntimeError(
                "GITHUB_PERSONAL_ACCESS_TOKEN is not set."
            )

        self.server_params = StdioServerParameters(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "-e",
                "GITHUB_READ_ONLY=1",
                "ghcr.io/github/github-mcp-server",
            ],
            env={
                **os.environ,
                "GITHUB_PERSONAL_ACCESS_TOKEN": token,
            },
        )

    @asynccontextmanager
    async def connect(self):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[dict]:
        async with self.connect() as session:
            result = await session.list_tools()

            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                }
                for tool in result.tools
            ]

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
    ):
        async with self.connect() as session:
            return await session.call_tool(
                name,
                arguments=arguments or {},
            )
