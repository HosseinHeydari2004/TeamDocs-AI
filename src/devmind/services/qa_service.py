
import asyncio
import logging

from devmind.agents.repo_qa import RepoQA
from devmind.github.tools import GitHubTools
from devmind.llm import get_chat_model
from src.rag.rag_pipeline import RagPipeline

logger = logging.getLogger(__name__)


class QAService:
    def __init__(self):
        self.rag = RagPipeline()
        self.repo_qa = RepoQA(get_chat_model())

        try:
            self.github = GitHubTools()
        except RuntimeError:
            logger.exception("GitHub MCP configuration failed")
            self.github = None

    async def answer(self, question: str, owner: str, repo: str):
        rag_context = ""
        github_context = ""
        warnings = []

        # 1. RAG retrieval
        try:
            result = await asyncio.to_thread(
                self.rag.run_pipeline,
                question,
            )

            messages = result.get("messages", [])

            for message in reversed(messages):
                if getattr(message, "type", "") == "ai":
                    content = getattr(message, "content", "")
                    if content and not getattr(message, "tool_calls", []):
                        rag_context = str(content)
                        break

            if not rag_context:
                warnings.append("RAG returned no final context.")

        except Exception:
            logger.exception("RAG retrieval failed")
            warnings.append("RAG retrieval failed")

        # 2. GitHub repository retrieval
        if self.github is not None:
            try:
                paths = [
                    "README.md",
                    "pyproject.toml",
                    "requirements.txt",
                ]

                results = await asyncio.gather(
                    *[
                        self.github.call_read_tool(
                            "get_file_contents",
                            {
                                "owner": owner,
                                "repo": repo,
                                "path": path,
                                "ref": "main",
                            },
                        )
                        for path in paths
                    ],
                    return_exceptions=True,
                )

                sections = []

                for path, result in zip(paths, results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Could not read GitHub file %s: %s",
                            path,
                            result,
                        )
                        continue

                    for item in result.content:
                        resource = getattr(item, "resource", None)
                        content = getattr(resource, "text", None)

                        if content:
                            sections.append(
                                f"FILE: {path}\n{content}"
                            )

                github_context = "\n\n".join(sections)

                if not github_context:
                    warnings.append("No GitHub files could be retrieved.")

            except Exception:
                logger.exception("GitHub retrieval failed")
                warnings.append("GitHub retrieval failed")

        else:
            warnings.append("GitHub MCP is not configured.")

        # 3. Generate final answer
        context = f"""
RAG CONTEXT:
{rag_context}

GITHUB CONTEXT:
{github_context}
"""

        if not rag_context and not github_context:
            return {
                "answer": "No repository context could be retrieved.",
                "rag_used": False,
                "github_used": False,
                "warnings": warnings,
            }

        answer = await asyncio.to_thread(
            self.repo_qa.answer,
            question,
            context,
        )

        return {
            "answer": answer,
            "rag_used": bool(rag_context),
            "github_used": bool(github_context),
            "warnings": warnings,
        }
