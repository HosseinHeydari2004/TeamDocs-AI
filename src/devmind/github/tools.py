
from devmind.github.mcp_client import GitHubMCPClient


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

    def __init__(self, client: GitHubMCPClient | None = None):
        self.client = client or GitHubMCPClient()

    async def list_tools(self) -> list[dict]:
        """Return only the tools allowed by this application."""
        tools = await self.client.list_tools()

        return [
            tool for tool in tools
            if tool["name"] in self.ALLOWED_TOOLS
        ]

    async def call_read_tool(
        self,
        name: str,
        arguments: dict | None = None,
    ):
        """Call an explicitly allowed tool."""
        if name not in self.ALLOWED_TOOLS:
            raise ValueError(
                f"Tool is not allowed by TeamDocs-AI: {name}"
            )

        return await self.client.call_tool(
            name,
            arguments or {},
        )
