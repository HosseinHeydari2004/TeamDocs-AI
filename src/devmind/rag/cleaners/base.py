from langchain_core.documents import Document
from abc import ABC, abstractmethod


class BaseCleaner(ABC):

    @abstractmethod
    def clean(self, text: str) -> list[Document]:
        raise NotImplementedError
