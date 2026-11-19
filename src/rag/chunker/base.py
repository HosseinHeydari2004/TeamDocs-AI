from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, docs: list[Document])-> list[Document]:
        raise NotImplementedError