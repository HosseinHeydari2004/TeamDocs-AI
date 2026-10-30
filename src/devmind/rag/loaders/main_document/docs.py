from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from pathlib import Path

from devmind.rag.loaders.base import BaseLoader


class DocsLoader(BaseLoader):

    def load(self, path: Path | str) -> list[Document]:
        loader = UnstructuredWordDocumentLoader(
            file_path=path,
            mode="elements",
            strategy="hi_res",
            infer_table_structure=True,
            language="en",
        )
        return loader.load()

