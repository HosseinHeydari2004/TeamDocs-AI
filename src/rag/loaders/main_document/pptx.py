from pathlib import Path

from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.documents import Document

from rag.loaders.base import BaseLoader


class PPTXLoader(BaseLoader):
    def load(self, path: Path | str) -> list[Document]:
        loader = UnstructuredPowerPointLoader(
            file_path=str(path),
            mode="paged",
            include_page_breaks=True,
        )
        return loader.load()
