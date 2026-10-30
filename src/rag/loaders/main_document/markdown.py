from rag.loaders.base import BaseLoader
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class MarkdownLoader(BaseLoader):

    def load(self, path: Path) -> list[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=path,
            mode="single",
        )
        return loader.load()
