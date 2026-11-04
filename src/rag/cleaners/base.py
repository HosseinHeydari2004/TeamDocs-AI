from langchain_core.documents import Document
from abc import ABC, abstractmethod


class BaseCleaner(ABC):

    @abstractmethod
    def clean(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError
