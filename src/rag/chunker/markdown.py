from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from .base import BaseChunker


class MarkdownChunker(BaseChunker):
    def __init__(self, ):
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=120
        )

    def chunk(self, docs: list[Document]) -> list[Document]:
        result = []
        for d in docs:
            header_docs = self._header_splitter.split_text(d.page_content)
            for hd in header_docs:
                hd.metadata = {**d.metadata, **hd.metadata}
            result.extend(self._splitter.split_documents(header_docs))
        return result