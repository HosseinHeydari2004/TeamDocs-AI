"""GitHub sub-agent: read-only repository / issue / pull-request analysis through the GitHub MCP server."""
from __future__ import annotations

import asyncio
import logging
import time

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from devmind.agents.common import final_answer, limits
from devmind.agents.contracts import GitHubRef, SubAgentError, SubAgentResult
from devmind.agents.errors import describe_failure
from devmind.agents.prompts import GITHUB_SUBAGENT_PROMPT
from devmind.github.errors import GitHubNotConfiguredError
from devmind.github.langchain_tools import GitHubRun, build_github_tools
from devmind.github.tools import GitHubTools
from devmind.llm import ModelManager

logger = logging.getLogger(__name__)


class GitHubSubAgent:
    name = "github"

    def __init__(self, models: ModelManager, github_factory=None) -> None:
        """``github_factory(settings) -> GitHubTools`` is injectable (tests); default uses the settings token."""
        self._models = models
        self._factory = github_factory or self._default_factory

    @staticmethod
    def _default_factory(settings) -> GitHubTools:
        token = settings.github_personal_access_token
        if token is None:
            raise GitHubNotConfiguredError(
                "GitHub is not configured: set GITHUB_PERSONAL_ACCESS_TOKEN in .env (a read-only token is enough)."
            )
        from devmind.github.mcp_client import GitHubMCPClient

        return GitHubTools(client=GitHubMCPClient(token=token.get_secret_value()))

    async def run(self, question: str, repository: str | None = None) -> SubAgentResult:
        started = time.perf_counter()
        settings = self._models.settings
        run = GitHubRun()
        try:
            github = self._factory(settings)  # fails fast (no LLM call) when no token is configured
        except Exception as exc:
            failure = describe_failure(exc)
            return SubAgentResult(agent="github", ok=False, error=SubAgentError(code=failure.code, message=failure.message),
                                  duration_ms=_ms(started))

        prompt = question if not repository else f"Repository: {repository}\n\n{question}"
        try:
            async with github.session() as session:
                tools = build_github_tools(session, run)

                async def attempt(handle):
                    agent = create_agent(
                        model=handle.model, tools=tools, system_prompt=GITHUB_SUBAGENT_PROMPT,
                        middleware=limits(settings.sub_agent_max_tool_calls),
                    )
                    return await asyncio.wait_for(
                        agent.ainvoke({"messages": [HumanMessage(content=prompt)]},
                                      config={"recursion_limit": settings.agent_recursion_limit}),
                        settings.agent_timeout_seconds,
                    )

                result, handle = await self._models.arun("github", attempt)
        except Exception as exc:
            failure = describe_failure(exc)
            return SubAgentResult(agent="github", ok=False, tool_calls=run.calls,
                                  error=SubAgentError(code=failure.code, message=failure.message), duration_ms=_ms(started))

        if run.calls and run.successes == 0:
            # Every GitHub call failed: report that, do not let a model-written answer pass as a result.
            code = "github_auth" if run.auth_failed else "github_error"
            message = ("GitHub rejected the configured token (bad or expired credentials)." if run.auth_failed
                       else f"All GitHub calls failed: {run.errors[0][:160]}")
            return SubAgentResult(agent="github", ok=False, model=handle.describe(), tool_calls=run.calls,
                                  error=SubAgentError(code=code, message=message), duration_ms=_ms(started))

        answer = final_answer(result)
        if not answer:
            return SubAgentResult(agent="github", ok=False, model=handle.describe(), tool_calls=run.calls,
                                  error=SubAgentError(code="agent_limit", message="The GitHub agent did not produce an answer within its limits."),
                                  duration_ms=_ms(started))
        refs, seen = [], set()
        for ref in run.refs:
            if ref["label"] not in seen:
                seen.add(ref["label"])
                refs.append(GitHubRef(**ref))
        return SubAgentResult(agent="github", ok=True, answer=answer, github=refs, model=handle.describe(),
                              tool_calls=run.calls, duration_ms=_ms(started))


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
