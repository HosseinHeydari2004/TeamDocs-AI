"""LEGACY (not used by the API): repository indexing flow that predates the Main Agent.

It builds its own RagPipeline/embedding model per repository. Repository questions are now answered live by the
GitHub sub-agent (POST /api/v1/chat). Kept for reference; see README 'Known limitations'.
"""
import asyncio
import hashlib
import logging
import shutil
from pathlib import Path

from devmind.agents.repo_qa import RepoQA
from devmind.github.repository_analyzer import RepositoryAnalyzer
from devmind.github.tools import GitHubTools
from devmind.llm import get_chat_model
from rag.rag_pipeline import RagPipeline

logger = logging.getLogger(__name__)


class QAService:
    REPOSITORY_CACHE = (
        Path.home() / ".teamdocs-ai" / "repositories"
    )

    rag_pipelines: dict[str, RagPipeline] = {}
    repo_locks: dict[str, asyncio.Lock] = {}

    def __init__(
            self,
            openrouter_api_key: str,
            github_token: str,
    ):
        chat_model = get_chat_model(api_key=openrouter_api_key)

        self.repo_qa = RepoQA(chat_model)
        self.rag_pipelines: dict[str, RagPipeline] = {}
        self.repo_locks: dict[str, asyncio.Lock] = {}

        self.github = GitHubTools(token=github_token)
        self.chat_model = chat_model

    @staticmethod
    def _repository_key(owner: str, repo: str) -> str:
        value = f"{owner.lower()}/{repo.lower()}"
        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()[:20]

    def _get_repository_lock(self, key: str) -> asyncio.Lock:
        if key not in self.repo_locks:
            self.repo_locks[key] = asyncio.Lock()

        return self.repo_locks[key]


    async def _get_rag_pipeline(self, key):
        pipeline_key = f"{key}_{id(self.chat_model)}"

        if pipeline_key not in self.rag_pipelines:
            pipeline = await asyncio.to_thread(
                RagPipeline,
                collection_name=f"repo_{key}",
                chat_model=self.chat_model,
            )
            self.rag_pipelines[pipeline_key] = pipeline

        return self.rag_pipelines[pipeline_key]

    async def _index_repository(
        self,
        owner: str,
        repo: str,
        key: str,
        rag: RagPipeline,
    ) -> dict:
        # Reuse an existing persistent repository index.
        existing_count = await asyncio.to_thread(
            rag._vectorstore.count
        )

        if existing_count > 0:
            logger.info(
                "Reusing repository index: %s/%s (%d chunks)",
                owner,
                repo,
                existing_count,
            )

            return {
                "status": "reused",
                "indexed_files": 0,
                "document_count": existing_count,
                "warnings": [],
            }

        if self.github is None:
            raise RuntimeError(
                "GitHub MCP is not configured."
            )

        analyzer = RepositoryAnalyzer(
            github=self.github,
        )

        # 1. Discover the repository structure.
        structure = await analyzer.discover_structure(
            owner=owner,
            repo=repo,
        )

        if not structure.get("files"):
            raise RuntimeError(
                "No repository files were discovered."
            )

        # 2. Prepare a stable local directory.
        destination = self.REPOSITORY_CACHE / key

        await asyncio.to_thread(
            shutil.rmtree,
            destination,
            ignore_errors=True,
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 3. Download files through GitHub MCP.
        download_result = (
            await analyzer.download_repository_files(
                owner=owner,
                repo=repo,
                structure=structure,
                destination=destination,
            )
        )

        downloaded_count = download_result["downloaded_count"]

        if downloaded_count == 0:
            raise RuntimeError(
                "No supported repository files could be downloaded."
            )

        logger.info(
            "Downloaded %d files from %s/%s",
            downloaded_count,
            owner,
            repo,
        )

        # 4. Send the local repository directory to RAG.
        await asyncio.to_thread(
            rag.add_documents,
            destination,
        )

        document_count = await asyncio.to_thread(
            rag._vectorstore.count
        )

        if document_count == 0:
            raise RuntimeError(
                "RAG indexing completed without storing documents."
            )

        warnings = []

        if structure.get("warnings"):
            warnings.extend(structure["warnings"])

        if download_result["failed_count"] > 0:
            warnings.append(
                f"{download_result['failed_count']} files could not be downloaded."
            )

        logger.info(
            "Repository indexed: %s/%s, chunks=%d",
            owner,
            repo,
            document_count,
        )

        return {
            "status": "indexed",
            "indexed_files": downloaded_count,
            "document_count": document_count,
            "warnings": warnings,
        }

    @staticmethod
    def _extract_retrieved_context(result) -> str:
        messages = result.get("messages", [])
        sections = []

        # Collect actual documents returned by RAG retrieval tools.
        for message in messages:
            if getattr(message, "type", "") != "tool":
                continue

            content = getattr(message, "content", "")

            if content and content.strip() != "No relevant documents found.":
                sections.append(str(content))

        return "\n\n".join(sections)

    async def answer(
        self,
        question: str,
        owner: str,
        repo: str,
    ):
        key = self._repository_key(owner, repo)

        # Prevent duplicate indexing of the same repository.
        lock = self._get_repository_lock(key)

        async with lock:
            try:
                # 1. Get the repository-specific RAG pipeline.
                rag = await self._get_rag_pipeline(key)

                # 2. Index the repository if it has not been indexed.
                index_info = await self._index_repository(
                    owner=owner,
                    repo=repo,
                    key=key,
                    rag=rag,
                )

                # 3. Retrieve relevant content from RAG.
                result = await asyncio.to_thread(
                    rag.run_pipeline,
                    question,
                )

                rag_context = self._extract_retrieved_context(
                    result
                )

                if not rag_context:
                    return {
                        "answer": (
                            "The repository was indexed, but RAG "
                            "did not retrieve relevant content for "
                            "this question."
                        ),
                        "repository": f"{owner}/{repo}",
                        "rag_used": False,
                        "github_used": True,
                        "index_status": index_info["status"],
                        "indexed_files": index_info["indexed_files"],
                        "document_count": index_info["document_count"],
                        "warnings": index_info["warnings"],
                    }

                # 4. Generate the final grounded answer with OpenRouter.
                answer = await asyncio.to_thread(
                    self.repo_qa.answer,
                    question,
                    rag_context,
                )

                return {
                    "answer": answer,
                    "repository": f"{owner}/{repo}",
                    "rag_used": True,
                    "github_used": True,
                    "index_status": index_info["status"],
                    "indexed_files": index_info["indexed_files"],
                    "document_count": index_info["document_count"],
                    "warnings": index_info["warnings"],
                }

            except Exception:
                logger.exception(
                    "Repository QA failed for %s/%s",
                    owner,
                    repo,
                )
                raise
