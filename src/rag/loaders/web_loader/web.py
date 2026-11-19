import os

os.environ.setdefault("USER_AGENT", "TeamDocs-AI/1.0")  # must be set before the LangChain web loader is imported

import logging

import bs4
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseLoader
from rag.loaders.web_loader.safe_http import SafeSession, WebFetchError, validate_public_url

logger = logging.getLogger(__name__)


class WebLoader(BaseLoader):
    """Download a page and extract its readable text (``article``/``main`` first, then generic text tags).

    Options (all optional): ``timeout`` (s), ``max_bytes``, ``max_redirects``, ``allow_private``.
    Failures raise :class:`WebFetchError` with a stable code instead of silently returning nothing.
    """

    MAIN_TAGS = ["article", "main"]
    TEXT_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table"]

    def _fetch(self, url: str, tags: list[str]) -> list[Document]:
        session = SafeSession(
            max_bytes=int(self.options.get("max_bytes", 5 * 1024 * 1024)),
            max_redirects=int(self.options.get("max_redirects", 5)),
            allow_private=bool(self.options.get("allow_private", False)),
        )
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"parse_only": bs4.SoupStrainer(tags)},
            session=session,
            requests_kwargs={"timeout": float(self.options.get("timeout", 15.0))},
            raise_for_status=True,
            show_progress=False,
        )
        try:
            return [d for d in loader.load() if d.page_content.strip()]
        except WebFetchError:
            raise
        except requests.exceptions.Timeout as exc:
            raise WebFetchError("timeout", "The website did not answer in time.") from exc
        except requests.exceptions.TooManyRedirects as exc:
            raise WebFetchError("too_many_redirects", "The URL redirected too many times.") from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "error"
            raise WebFetchError("http_status", f"The website answered HTTP {status}.") from exc
        except requests.exceptions.RequestException as exc:
            raise WebFetchError("network", f"The page could not be downloaded ({type(exc).__name__}).") from exc

    def load(self, url: str) -> list[Document]:
        urls = [url] if isinstance(url, str) else list(url)

        docs: list[Document] = []
        for u in urls:
            u = validate_public_url(u, allow_private=bool(self.options.get("allow_private", False)))
            result = self._fetch(u, self.MAIN_TAGS)
            if not result:
                result = self._fetch(u, self.TEXT_TAGS)

            if not result:
                logger.warning("Empty content: %s", u)

            for doc in result:
                doc.metadata["source"] = u
                doc.metadata["source_url"] = u
                doc.metadata["file_type"] = "web"
            docs.extend(result)

        return docs
