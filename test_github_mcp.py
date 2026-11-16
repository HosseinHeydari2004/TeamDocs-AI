
import asyncio
from devmind.github.tools import GitHubTools

async def main():
    github = GitHubTools()

    result = await github.call_read_tool(
        "get_file_contents",
        {
            "owner": "HosseinHeydari2004",
            "repo": "TeamDocs-AI",
            "path": "README.md",
            "ref": "main"
        },
    )

    for item in result.content:
        print(getattr(item, "text", str(item)))

asyncio.run(main())
