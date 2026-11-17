import re
import json
import logging

from collections import deque

from devmind.github.tools import GitHubTools

from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)


class RepositoryAnalyzer:
    """Discover a GitHub repository structure dynamically."""

    SKIP_DIRS = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "vendor",
        "coverage",
        ".idea",
        ".next",
        "target",
    }

    SOURCE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".h", ".cs", ".php", ".rb", ".swift",
        ".kt", ".scala", ".sh", ".sql", ".ipynb",
    }

    DOC_EXTENSIONS = {
        ".md", ".rst", ".txt", ".adoc", ".mdx",
    }

    MANIFEST_FILES = {
        "pyproject.toml",
        "requirements.txt",
        "pipfile",
        "setup.py",
        "package.json",
        "cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "composer.json",
        "gemfile",
    }

    STOP_WORDS = {
        "the", "and", "for", "with", "from",
        "what", "how", "where", "which",
        "this", "that", "does", "are",
        "is", "of", "in", "to", "a", "an",
        "را", "از", "به", "در", "و", "برای",
    }

    @classmethod
    def _tokens(cls, value: str) -> set[str]:
        tokens = re.findall(
            r"[a-zA-Z0-9_]+|[\u0600-\u06FF]+",
            value.lower(),
        )

        return {
            token
            for token in tokens
            if len(token) > 1 and token not in cls.STOP_WORDS
        }

    @classmethod
    def select_relevant_files(
        cls,
        structure: dict,
        question: str,
        limit: int = 12,
    ) -> list[dict]:
        """Rank discovered repository files by relevance to a question."""

        question_tokens = cls._tokens(question)
        question_lower = question.lower()

        architecture_terms = {
            "architecture", "structure", "overview",
            "design", "component", "workflow",
            "معماری", "ساختار", "اجزا",
        }

        setup_terms = {
            "install", "installation", "setup",
            "dependency", "dependencies",
            "نصب", "وابستگی",
        }

        test_terms = {
            "test", "testing", "tests",
            "آزمون", "تست",
        }

        architecture_question = bool(
            question_tokens & architecture_terms
        )
        setup_question = bool(question_tokens & setup_terms)
        test_question = bool(question_tokens & test_terms)

        ranked_files = []

        for file in structure.get("files", []):
            path = file.get("path", "").replace("\\", "/")
            name = file.get("name", "")
            basename = name.lower()
            extension = (
                "." + basename.rsplit(".", 1)[-1]
                if "." in basename else ""
            )

            path_tokens = cls._tokens(path)
            name_tokens = cls._tokens(name)

            score = 0
            reasons = []

            # Match words from the question with the filename.
            name_matches = question_tokens & name_tokens
            if name_matches:
                score += 5 * len(name_matches)
                reasons.append("filename matches question")

            # Match words from the question with the full path.
            path_matches = question_tokens & path_tokens
            if path_matches:
                score += 2 * len(path_matches)
                reasons.append("path matches question")

            # Exact phrase match.
            if question_lower.strip() in path.lower():
                score += 8
                reasons.append("exact phrase match")

            # General programming source files.
            if extension in cls.SOURCE_EXTENSIONS:
                score += 1

            # Documentation files.
            if extension in cls.DOC_EXTENSIONS:
                score += 2

            # Conventional repository overview files.
            if basename.startswith("readme"):
                score += 5
                reasons.append("repository documentation")

            # Architecture questions benefit from docs and source files.
            if architecture_question:
                if any(
                    part in {"docs", "doc", "documentation"}
                    for part in path.lower().split("/")
                ):
                    score += 5
                    reasons.append("documentation directory")

                if basename in cls.MANIFEST_FILES:
                    score += 4
                    reasons.append("project manifest")

                if extension in cls.SOURCE_EXTENSIONS:
                    score += 2
                    reasons.append("source code")

            # Setup questions benefit from dependency manifests.
            if setup_question and basename in cls.MANIFEST_FILES:
                score += 8
                reasons.append("dependency or build configuration")

            # Test questions benefit from test-related files.
            if test_question:
                if (
                    "test" in path.lower().split("/")
                    or basename.startswith("test")
                    or "_test" in basename
                ):
                    score += 8
                    reasons.append("test-related file")

            ranked_files.append({
                **file,
                "score": score,
                "reasons": reasons,
            })

        # If there are no textual matches, use a general fallback.
        if ranked_files and all(item["score"] == 0 for item in ranked_files):
            for item in ranked_files:
                path = item["path"].lower()
                extension = (
                    "." + item["name"].rsplit(".", 1)[-1].lower()
                    if "." in item["name"] else ""
                )

                if item["name"].lower().startswith("readme"):
                    item["score"] += 5
                elif extension in cls.DOC_EXTENSIONS:
                    item["score"] += 3
                elif extension in cls.SOURCE_EXTENSIONS:
                    item["score"] += 2
                elif item["name"].lower() in cls.MANIFEST_FILES:
                    item["score"] += 4

        ranked_files.sort(
            key=lambda item: (-item["score"], item["path"].lower())
        )

        return ranked_files[:limit]

    def __init__(
        self,
        github: GitHubTools | None = None,
        max_depth: int = 5,
        max_directories: int = 50,
        max_files: int = 1000,
    ):
        self.github = github or GitHubTools()
        self.max_depth = max_depth
        self.max_directories = max_directories
        self.max_files = max_files

    @staticmethod
    def _parse_entries(result) -> list[dict]:
        """Parse directory listing returned by GitHub MCP."""

        for item in result.content:
            resource = getattr(item, "resource", None)
            text = getattr(resource, "text", None)

            if not text:
                text = getattr(item, "text", None)

            if not text:
                continue

            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue

            if isinstance(data, list):
                return [
                    entry for entry in data
                    if isinstance(entry, dict)
                ]

        return []

    async def discover_structure(
        self,
        owner: str,
        repo: str,
    ) -> dict:
        """Recursively discover repository files and directories."""

        queue = deque([("", 0)])
        visited = set()
        directories = []
        files = []
        warnings = []

        while queue:
            path, depth = queue.popleft()

            if path in visited:
                continue

            if len(visited) >= self.max_directories:
                warnings.append("Directory scan limit reached.")
                break

            visited.add(path)

            try:
                result = await self.github.call_read_tool(
                    "get_file_contents",
                    {
                        "owner": owner,
                        "repo": repo,
                        "path": path,
                    },
                )

                entries = self._parse_entries(result)

                for entry in entries:
                    entry_type = entry.get("type")
                    entry_path = entry.get("path", "")
                    name = entry.get("name", "")

                    if not entry_path:
                        continue

                    if entry_type == "dir":
                        directories.append(entry_path)

                        if (
                            depth < self.max_depth
                            and name not in self.SKIP_DIRS
                        ):
                            queue.append((entry_path, depth + 1))

                    elif entry_type == "file":
                        files.append({
                            "name": name,
                            "path": entry_path,
                            "size": entry.get("size", 0),
                        })

                        if len(files) >= self.max_files:
                            warnings.append("File scan limit reached.")
                            queue.clear()
                            break

            except Exception as exc:
                logger.exception(
                    "Failed to inspect directory: %s",
                    path or "/",
                )
                warnings.append(
                    f"Could not inspect {path or '/'}: {type(exc).__name__}"
                )

        return {
            "repository": f"{owner}/{repo}",
            "directories_scanned": len(visited),
            "directory_count": len(directories),
            "file_count": len(files),
            "directories": directories,
            "files": files,
            "warnings": warnings,
        }


    async def download_repository_files(
        self,
        owner: str,
        repo: str,
        structure: dict,
        destination: str | Path,
        max_file_size: int = 1_000_000,
    ) -> dict:
        """Download supported text files while preserving repository paths."""

        supported_extensions = {
            ".py", ".js", ".jsx", ".mjs", ".cjs",
            ".ts", ".tsx", ".java", ".go", ".rs",
            ".cpp", ".cc", ".cxx", ".h", ".hpp",
            ".cs", ".kt", ".kts", ".sql",
            ".html", ".htm", ".md", ".txt",
        }

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        candidates = [
            file for file in structure.get("files", [])
            if Path(file["path"]).suffix.lower() in supported_extensions
            and file.get("size", 0) <= max_file_size
        ]

        calls = [
            (
                "get_file_contents",
                {
                    "owner": owner,
                    "repo": repo,
                    "path": file["path"],
                },
            )
            for file in candidates
        ]

        results = await self.github.call_read_tools(calls)

        downloaded = []
        failed = []

        for file, result in zip(candidates, results):
            if isinstance(result, Exception) or getattr(result, "isError", False):
                failed.append(file["path"])
                continue

            content_parts = []

            for item in result.content:
                resource = getattr(item, "resource", None)
                content = getattr(resource, "text", None)

                if content is None:
                    content = getattr(item, "text", None)

                if content:
                    content_parts.append(content)

            if not content_parts:
                failed.append(file["path"])
                continue

            content = "\n".join(content_parts)

            relative_path = PurePosixPath(file["path"])

            relative_path = PurePosixPath(file["path"])

            if (
                    relative_path.is_absolute()
                    or not relative_path.parts
                    or ".." in relative_path.parts
                    or ":" in relative_path.parts[0]
            ):
                failed.append(file["path"])
                continue

            local_path = destination.joinpath(*relative_path.parts).resolve()

            if not local_path.is_relative_to(destination.resolve()):
                failed.append(file["path"])
                continue

            try:
                local_path.write_text(content, encoding="utf-8")
                downloaded.append(file["path"])
            except OSError:
                failed.append(file["path"])

        return {
            "downloaded_count": len(downloaded),
            "failed_count": len(failed),
            "downloaded_files": downloaded,
            "failed_files": failed,
            "destination": str(destination),
        }

