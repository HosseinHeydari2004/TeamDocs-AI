from devmind.rag.loaders.base import BaseLoader
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class MarkdownLoader(BaseLoader):

    def load(self, path: Path) -> list[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=path,
            mode="elements",
            unstructured_kwargs={
                "chunking_strategy": "by_title",
                "max_characters": 1000,
                "combine_text_under_n_chars": 200,
                "new_after_n_chars": 800,
            },
        )
        return loader.load()

