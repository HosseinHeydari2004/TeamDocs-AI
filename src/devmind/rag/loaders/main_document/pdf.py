from devmind.rag.loaders.base import BaseLoader
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader


class PDFLoader(BaseLoader):

    def load(self, path: Path | str) -> list[Document]:
        loader = UnstructuredPDFLoader(
            file_path=path,
            mode="elements",
            strategy="hi_res",
            language=["eng"],
            infer_table_structure=True
        )
        return loader.load()

