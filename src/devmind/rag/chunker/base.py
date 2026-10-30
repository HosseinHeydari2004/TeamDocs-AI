from langchain_core.documents import Document
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: list[Document])-> list[Document]:
        raise NotImplementedError