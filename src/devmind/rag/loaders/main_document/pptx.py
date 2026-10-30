from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from pathlib import Path
from devmind.rag.loaders.base import BaseLoader


class PPTXLoader(BaseLoader):

    def load(self, path: Path | str) -> list[Document]:
        loader = UnstructuredPowerPointLoader(
            file_path=path,
            mode="paged",
            strategy="hi_res",
            languages=["eng"],
            include_page_breaks=True
        )

