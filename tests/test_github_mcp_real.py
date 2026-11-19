"""Integration with the REAL official GitHub MCP server process (stdio) against a local fake GitHub API.

Real: the ``github-mcp-server`` binary, the MCP protocol/SDK, our MCP client, GitHubTools, RepositoryAnalyzer and
the GitHub sub-agent. Fake: GitHub's REST API (a local HTTP server, reached through the server's ``--gh-host``
option) and the language model (scripted). Not covered: a real token against github.com.

Run with:  GITHUB_MCP_BINARY=/path/to/github-mcp-server pytest tests/test_github_mcp_real.py
(binary: https://github.com/github/github-mcp-server/releases). Skipped when the variable is not set.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from devmind.agents.github_agent import GitHubSubAgent
from devmind.github.errors import GitHubNotConfiguredError, GitHubUnavailableError
from devmind.github.mcp_client import GitHubMCPClient
from devmind.github.repository_analyzer import RepositoryAnalyzer
from devmind.github.results import result_is_error, result_text
from devmind.github.tools import GitHubTools
from devmind.llm import ModelManager

from .helpers import ScriptedToolModel, supervisor_policy

BINARY = os.environ.get("GITHUB_MCP_BINARY", "")
pytestmark = pytest.mark.skipif(not (BINARY and Path(BINARY).is_file()), reason="set GITHUB_MCP_BINARY to run")

README = "# Hello-World\nMy first repository on GitHub!\nRetries use exponential backoff.\n"
SRC = "def retry():\n    return 'backoff'\n"
TREE = {
    "README.md": README,
    "src/app.py": SRC,
}


def _sha(text: str) -> str:
    data = text.encode()  # real git blob hash: the MCP server verifies it
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


BLOBS = {_sha(body): body for body in TREE.values()}


class FakeGitHub(BaseHTTPRequestHandler):
    log: list[tuple[str, str | None]] = []

    def _send(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        FakeGitHub.log.append((self.path, self.headers.get("Authorization")))
        path = self.path.split("?")[0].removeprefix("/api/v3")
        repo = "/repos/octocat/Hello-World"
        if path == repo:
            return self._send(200, {"id": 1, "name": "Hello-World", "full_name": "octocat/Hello-World", "default_branch": "master", "owner": {"login": "octocat"}})
        if path.startswith(repo + "/git/ref/") or path.startswith(repo + "/git/refs/"):
            return self._send(200, {"ref": "refs/heads/master", "object": {"sha": "d" * 40, "type": "commit"}})
        if path.startswith(repo + "/git/blobs/"):
            body = BLOBS.get(path.rsplit("/", 1)[-1])
            if body is not None:
                return self._send(200, {"sha": _sha(body), "size": len(body), "encoding": "base64", "content": base64.b64encode(body.encode()).decode()})
        if path.startswith(repo + "/contents"):
            sub = path[len(repo + "/contents"):].strip("/")
            if sub in TREE:
                body = TREE[sub]
                return self._send(200, {"type": "file", "name": sub.split("/")[-1], "path": sub, "sha": _sha(body), "size": len(body), "encoding": "base64", "content": base64.b64encode(body.encode()).decode()})
            children = {}
            for name, body in TREE.items():
                if sub == "" or name.startswith(sub + "/"):
                    rest = name[len(sub) + 1:] if sub else name
                    head = rest.split("/")[0]
                    children[head] = {"type": "dir" if "/" in rest else "file", "name": head, "path": f"{sub}/{head}".lstrip("/"), "sha": _sha(body) if "/" not in rest else "b" * 40, "size": len(body) if "/" not in rest else 0}
            if children:
                return self._send(200, list(children.values()))
        return self._send(404, {"message": "Not Found"})

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def fake_api():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def client_for(fake_api, token="ghp_localfaketoken"):
    return GitHubMCPClient(token=token, command=BINARY, extra_args=["--gh-host", fake_api], timeout_seconds=30)


@pytest.mark.anyio
async def test_only_allowed_read_tools_are_exposed(fake_api):
    names = {t["name"] for t in await GitHubTools(client=client_for(fake_api)).list_tools()}
    assert {"get_file_contents", "list_issues", "pull_request_read", "search_code"} <= names
    assert not any(n.startswith(("create_", "update_", "delete_", "merge_")) for n in names)


@pytest.mark.anyio
async def test_read_file_and_directory_and_token_is_sent_as_header(fake_api):
    FakeGitHub.log.clear()
    tools = GitHubTools(client=client_for(fake_api, "ghp_localfaketoken"))
    async with tools.session() as gh:
        file_result = await gh.call("get_file_contents", {"owner": "octocat", "repo": "Hello-World", "path": "README.md"})
        assert not result_is_error(file_result) and "exponential backoff" in result_text(file_result)
        missing = await gh.call("get_file_contents", {"owner": "octocat", "repo": "Nope", "path": "x"})
        assert result_is_error(missing)  # mcp 2.x ``is_error`` really is what the server sets
    assert any(auth == "Bearer ghp_localfaketoken" for _p, auth in FakeGitHub.log)


@pytest.mark.anyio
async def test_write_tools_cannot_be_called_through_our_layer(fake_api):
    async with GitHubTools(client=client_for(fake_api)).session() as gh:
        with pytest.raises(ValueError, match="not allowed"):
            await gh.call("create_issue", {"owner": "o", "repo": "r", "title": "x"})


@pytest.mark.anyio
async def test_repository_analyzer_parses_real_server_output_and_detects_errors(fake_api, tmp_path):
    tools = GitHubTools(client=client_for(fake_api))
    analyzer = RepositoryAnalyzer(github=tools)
    structure = await analyzer.discover_structure("octocat", "Hello-World")
    assert {f["path"] for f in structure["files"]} >= {"README.md", "src/app.py"}, structure
    downloaded = await analyzer.download_repository_files("octocat", "Hello-World", structure, tmp_path)
    assert downloaded["failed_count"] == 0 and "exponential backoff" in (tmp_path / "README.md").read_text()

    # A repository that errors must be reported as failed, never written as file content (is_error regression).
    bad = await analyzer.download_repository_files("octocat", "Nope", {"files": [{"path": "a.py", "size": 10}]}, tmp_path / "bad")
    assert bad["downloaded_count"] == 0 and bad["failed_files"] == ["a.py"]
    assert not (tmp_path / "bad" / "a.py").exists()


@pytest.mark.anyio
async def test_github_subagent_end_to_end_with_the_real_mcp_server(fake_api, settings):
    model = ScriptedToolModel(policy=supervisor_policy(), calls=[], seen_tools=[])
    models = ModelManager(settings, fixed_model=model)
    agent = GitHubSubAgent(models, github_factory=lambda s: GitHubTools(client=client_for(fake_api)))
    result = await agent.run("Summarise the README", "octocat/Hello-World")
    assert result.ok and "Hello-World" in result.answer
    assert result.github[0].label == "octocat/Hello-World:README.md" and result.tool_calls == ["get_file_contents"]


@pytest.mark.anyio
async def test_all_calls_failing_is_reported_as_failure(fake_api, settings):
    def policy(messages, tools):
        from langchain_core.messages import AIMessage

        from .helpers import tool_call
        if any(m.type == "tool" for m in messages[1:]):
            return AIMessage(content="Everything is fine!")  # a model that ignores the failure
        return AIMessage(content="", tool_calls=[tool_call("github_get_file", {"owner": "octocat", "repo": "Nope", "path": "x"})])

    models = ModelManager(settings, fixed_model=ScriptedToolModel(policy=policy, calls=[], seen_tools=[]))
    agent = GitHubSubAgent(models, github_factory=lambda s: GitHubTools(client=client_for(fake_api)))
    result = await agent.run("read x", "octocat/Nope")
    assert result.ok is False and result.error.code == "github_error" and result.answer is None


def test_client_errors_are_clear_and_never_contain_the_token(tmp_path):
    with pytest.raises(GitHubNotConfiguredError):
        GitHubMCPClient(token="", command=BINARY)
    client = GitHubMCPClient(token="ghp_SECRETTOKEN", command=str(tmp_path / "missing-binary"))

    async def run():
        async with client.connect():
            pass

    import asyncio

    with pytest.raises(GitHubUnavailableError) as info:
        asyncio.run(run())
    assert "ghp_SECRETTOKEN" not in str(info.value) and "GITHUB_MCP_COMMAND" in str(info.value)
