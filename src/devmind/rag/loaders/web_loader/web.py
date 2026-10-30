import os

os.environ.setdefault("USER_AGENT", "TeamDocs-AI/1.0")  # ظ‚ط¨ظ„ ط§ط² import ظ„ظˆط¯ط± ظ„ظ†ع¯â€Œع†غŒظ†

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from devmind.rag.loaders.base import BaseLoader


class WebLoader(BaseLoader):
    MAIN_TAGS = ["article", "main"]
    TEXT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table"]

    def _fetch(self, url: str, tags: list[str]) -> list[Document]:
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"parse_only": bs4.SoupStrainer(tags)},
            continue_on_failure=True,
        )
        return [d for d in loader.load() if d.page_content.strip()]

    def load(self, url: str) -> list[Document]:
        urls = [url] if isinstance(url, str) else list(url)

        docs: list[Document] = []
        for u in urls:
            result = self._fetch(u, self.MAIN_TAGS)
            if not result:
                result = self._fetch(u, self.TEXT_TAGS)

            if not result:
                print(f"Empty content: {u}")

            docs.extend(result)

        return docs

