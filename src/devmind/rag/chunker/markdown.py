from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import MarkdownNodeParser


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, docs: list[Document]) -> list[Document]:
        raise NotImplementedError


class MarkdownChunker(BaseChunker):
    def __init__(self, max_chars: int = 1200, overlap: int = 100) -> None:
        self.max_chars = max_chars
        self.overlap = overlap
        self._parser = MarkdownNodeParser()
        self._rec = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n```\n", "\n\n", "\n", " "],
            keep_separator=True,
        )

    def chunk(self, docs: list[Document]) -> list[Document]:
        out: list[Document] = []
        for d in docs:
            out.extend(self._chunk_one(d))
        return out


    def _chunk_one(self, d: Document) -> list[Document]:
        source = d.metadata.get("source", "")
        base_meta = {k: v for k, v in d.metadata.items() if k != "header_path"}

        nodes = self._parser.get_nodes_from_documents(
            [LIDocument(text=d.page_content)]
        )

        result: list[Document] = []
        for n in nodes:
            header_path = n.metadata.get("header_path", "")
            meta = {
                **base_meta,
                "source": source,
                "header_path": header_path,
            }

            text = n.text
            if len(text) <= self.max_chars:
                result.append(Document(page_content=text, metadata=meta))
            else:
                for sub in self._rec.split_text(text):
                    result.append(Document(page_content=sub, metadata={**meta}))

        return result