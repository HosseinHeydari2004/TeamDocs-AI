from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseCleaner(ABC):

    @abstractmethod
    def clean(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError
