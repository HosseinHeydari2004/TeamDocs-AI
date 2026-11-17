import asyncio

from devmind.github.tools import GitHubTools
from devmind.github.repository_analyzer import RepositoryAnalyzer


async def main():
    github = GitHubTools()
    analyzer = RepositoryAnalyzer(github=github)

    owner = "HosseinHeydari2004"
    repo = "TeamDocs-AI"

    # Discover repository structure
    structure = await analyzer.discover_structure(
        owner=owner,
        repo=repo,
    )

    print(f"\nRepository: {structure['repository']}")
    print(f"Directories: {structure['directory_count']}")
    print(f"Files: {structure['file_count']}")

    # Simulate a question from a user
    question = "Explain the architecture of this repository"

    # Select relevant files dynamically
    selected_files = analyzer.select_relevant_files(
        structure=structure,
        question=question,
        limit=12,
    )

    print(f"\nUser question: {question}")
    print("\nSelected relevant files:")

    for file in selected_files:
        print(f"\n- {file['path']}")
        print(f"  Score: {file['score']}")
        print(f"  Reasons: {', '.join(file['reasons'])}")


if __name__ == "__main__":
    asyncio.run(main())