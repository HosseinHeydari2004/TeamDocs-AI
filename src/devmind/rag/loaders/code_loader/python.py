from pathlib import Path
from langchain_community.document_loaders import PythonLoader
from langchain_core.documents import Document
from devmind.rag.loaders.base import BaseLoader

class PythonCodeLoader(BaseLoader):

    def load(self, path: Path | str) -> list[Document]:
        loader = PythonLoader(file_path=path)
        return loader.load()
