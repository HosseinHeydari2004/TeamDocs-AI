import asyncio
import json
import tempfile
from pathlib import Path

from devmind.github.tools import GitHubTools
from devmind.github.repository_analyzer import RepositoryAnalyzer


OWNER = "HosseinHeydari2004"
REPO = "TeamDocs-AI"

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".h",
    ".hpp", ".cs", ".kt", ".sql",
    ".html", ".htm", ".md", ".txt",
}


async def main():
    github = GitHubTools()
    analyzer = RepositoryAnalyzer(github=github)

    print("Discovering repository structure...")

    structure = await analyzer.discover_structure(
        owner=OWNER,
        repo=REPO,
    )

    files = [
        file for file in structure.get("files", [])
        if Path(file["path"]).suffix.lower() in SUPPORTED_EXTENSIONS
        and file.get("size", 0) <= 1_000_000
    ]

    print(f"Discovered files: {len(files)}")

    if not files:
        raise RuntimeError("No supported text files found.")

    # Download only the first 3 files for this test.
    test_structure = {
        **structure,
        "files": files[:3],
    }

    with tempfile.TemporaryDirectory(prefix="teamdocs-github-") as temp_dir:
        print("Downloading test files...")

        result = await analyzer.download_repository_files(
            owner=OWNER,
            repo=REPO,
            structure=test_structure,
            destination=temp_dir,
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))

        downloaded = result["downloaded_count"]

        if downloaded == 0:
            raise RuntimeError("No files were downloaded successfully.")

        print("\nVerified files on disk:")

        for relative_path in result["downloaded_files"]:
            local_file = Path(temp_dir).joinpath(*Path(relative_path).parts)
            print(f"- {relative_path}: {local_file.stat().st_size} bytes")

        print("\nGitHub download test passed.")


if __name__ == "__main__":
    asyncio.run(main())