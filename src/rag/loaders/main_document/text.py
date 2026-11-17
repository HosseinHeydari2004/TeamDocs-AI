from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from rag.loaders.base import BaseLoader

class TXTLoader(BaseLoader):
    def load(self, path: Path | str) -> list[Document]:
        loader = TextLoader(file_path=str(path))
        return loader.load()