"""GitHub-layer exceptions. Messages are safe to show to users (they never contain the token)."""
from __future__ import annotations


class GitHubError(RuntimeError):
    """Base class for GitHub integration failures."""


class GitHubNotConfiguredError(GitHubError):
    """No GitHub credential is configured (GITHUB_PERSONAL_ACCESS_TOKEN)."""


class GitHubUnavailableError(GitHubError):
    """The GitHub MCP server could not be started or did not answer (Docker missing, timeout...)."""


class GitHubAuthError(GitHubError):
    """GitHub rejected the configured token."""
