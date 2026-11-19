"""LangChain tools over the read-only GitHub MCP server, for the GitHub sub-agent.

Arguments are validated before they reach GitHub, outputs are truncated, and every call is
recorded (``GitHubRun``) so the API can report exactly which GitHub objects were read and
whether any call failed. Repository content is untrusted data: it is returned to the model as a
tool result, never as instructions.
"""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import BaseTool, tool

from devmind.github.results import result_is_error, result_text

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_PATH = re.compile(r"^[^\x00-\x1f]{0,500}$")
MAX_TOOL_OUTPUT_CHARS = 6000


class GitHubArgumentError(ValueError):
    pass


@dataclass
class GitHubRun:
    """Everything one GitHub sub-agent run did (for tracing and honest status reporting)."""

    calls: list[str] = field(default_factory=list)
    refs: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    successes: int = 0

    @property
    def auth_failed(self) -> bool:
        return any("401" in e or "bad credentials" in e.lower() or "requires authentication" in e.lower() for e in self.errors)


def _check_repo(owner: str, repo: str) -> None:
    if not _NAME.match(owner or "") or not _NAME.match(repo or ""):
        raise GitHubArgumentError("owner and repo must be plain GitHub names such as 'octocat' and 'Hello-World'.")


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text or "(empty result)"
    return text[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [truncated, {len(text) - MAX_TOOL_OUTPUT_CHARS} more characters]"


def build_github_tools(session, run: GitHubRun) -> list[BaseTool]:
    """Create the tools bound to an open ``GitHubSession`` and a ``GitHubRun`` recorder."""

    async def call(name: str, arguments: dict, ref: dict | None) -> str:
        run.calls.append(name)
        try:
            result = await session.call(name, arguments)
        except Exception as exc:  # timeout, broken pipe, server crash ...
            message = f"GitHub call failed: {type(exc).__name__}"
            run.errors.append(message)
            logger.warning("GitHub tool %s failed: %s", name, type(exc).__name__)
            return f"ERROR: {message}"
        text = result_text(result)
        if result_is_error(result):
            run.errors.append(text[:300] or "GitHub returned an error")
            return f"ERROR: {_clip(text)}"
        run.successes += 1
        if ref:
            run.refs.append(ref)
        return _clip(text)

    def guarded(func):
        @functools.wraps(func)  # keeps the real signature, which LangChain turns into the tool schema
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except GitHubArgumentError as exc:
                run.errors.append(str(exc))
                return f"ERROR: {exc}"

        return wrapper

    @tool
    @guarded
    async def github_get_file(owner: str, repo: str, path: str = "", ref: str | None = None) -> str:
        """Read a file, or list a directory, in a GitHub repository. Use path='' for the repository root
        (then read README.md or manifests such as pyproject.toml / package.json to understand a project)."""
        _check_repo(owner, repo)
        if not _PATH.match(path or "") or ".." in (path or "").split("/"):
            raise GitHubArgumentError("path must be a relative path inside the repository.")
        args = {"owner": owner, "repo": repo, "path": path or ""}
        if ref:
            args["ref"] = ref
        label = f"{owner}/{repo}:{path or '/'}"
        return await call("get_file_contents", args, {
            "kind": "directory" if not path or path.endswith("/") else "file",
            "repository": f"{owner}/{repo}", "label": label,
            "url": f"https://github.com/{owner}/{repo}/blob/{ref or 'HEAD'}/{path}" if path else f"https://github.com/{owner}/{repo}",
        })

    @tool
    @guarded
    async def github_search_code(query: str, owner: str | None = None, repo: str | None = None) -> str:
        """Search code on GitHub. Give owner and repo to restrict the search to one repository."""
        if not query or len(query) > 256:
            raise GitHubArgumentError("query must be 1-256 characters.")
        scope = ""
        if owner and repo:
            _check_repo(owner, repo)
            scope = f" repo:{owner}/{repo}"
        return await call("search_code", {"query": query + scope, "perPage": 10}, {
            "kind": "code_search", "repository": f"{owner}/{repo}" if owner and repo else None, "label": f"code search: {query}",
        })

    @tool
    @guarded
    async def github_list_issues(owner: str, repo: str, state: Literal["OPEN", "CLOSED"] = "OPEN") -> str:
        """List issues of a repository (most recent first)."""
        _check_repo(owner, repo)
        return await call("list_issues", {"owner": owner, "repo": repo, "state": state, "perPage": 20}, {
            "kind": "issue", "repository": f"{owner}/{repo}", "label": f"{owner}/{repo} issues ({state.lower()})",
            "url": f"https://github.com/{owner}/{repo}/issues",
        })

    @tool
    @guarded
    async def github_get_issue(
        owner: str, repo: str, issue_number: int, include: Literal["details", "comments"] = "details"
    ) -> str:
        """Read one issue (details) or its comments."""
        _check_repo(owner, repo)
        method = "get" if include == "details" else "get_comments"
        return await call("issue_read", {"method": method, "owner": owner, "repo": repo, "issue_number": issue_number}, {
            "kind": "issue", "repository": f"{owner}/{repo}", "label": f"{owner}/{repo}#{issue_number}",
            "url": f"https://github.com/{owner}/{repo}/issues/{issue_number}",
        })

    @tool
    @guarded
    async def github_list_pull_requests(
        owner: str, repo: str, state: Literal["open", "closed", "all"] = "open"
    ) -> str:
        """List pull requests of a repository."""
        _check_repo(owner, repo)
        return await call("list_pull_requests", {"owner": owner, "repo": repo, "state": state, "perPage": 20}, {
            "kind": "pull_request", "repository": f"{owner}/{repo}", "label": f"{owner}/{repo} pull requests ({state})",
            "url": f"https://github.com/{owner}/{repo}/pulls",
        })

    @tool
    @guarded
    async def github_get_pull_request(
        owner: str, repo: str, pull_number: int,
        include: Literal["details", "files", "diff", "reviews", "comments"] = "details",
    ) -> str:
        """Read one pull request: details, changed files, diff, reviews or comments."""
        _check_repo(owner, repo)
        method = {"details": "get", "files": "get_files", "diff": "get_diff", "reviews": "get_reviews", "comments": "get_comments"}[include]
        return await call("pull_request_read", {"method": method, "owner": owner, "repo": repo, "pullNumber": pull_number}, {
            "kind": "pull_request", "repository": f"{owner}/{repo}", "label": f"{owner}/{repo}!{pull_number} ({include})",
            "url": f"https://github.com/{owner}/{repo}/pull/{pull_number}",
        })

    @tool
    @guarded
    async def github_list_commits(owner: str, repo: str, path: str | None = None) -> str:
        """List recent commits of a repository, optionally only those touching one path."""
        _check_repo(owner, repo)
        args: dict = {"owner": owner, "repo": repo, "perPage": 15}
        if path:
            args["path"] = path
        return await call("list_commits", args, {
            "kind": "commits", "repository": f"{owner}/{repo}", "label": f"{owner}/{repo} commits",
            "url": f"https://github.com/{owner}/{repo}/commits",
        })

    @tool
    @guarded
    async def github_search_repositories(query: str) -> str:
        """Search public GitHub repositories (use when the user did not give an exact owner/repo)."""
        if not query or len(query) > 256:
            raise GitHubArgumentError("query must be 1-256 characters.")
        return await call("search_repositories", {"query": query, "perPage": 5}, {
            "kind": "repositories", "repository": None, "label": f"repository search: {query}",
        })

    return [
        github_get_file, github_search_code, github_list_issues, github_get_issue,
        github_list_pull_requests, github_get_pull_request, github_list_commits, github_search_repositories,
    ]
