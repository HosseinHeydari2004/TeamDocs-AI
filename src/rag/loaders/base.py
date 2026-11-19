from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseLoader(ABC):
    """Base class for all loaders.

    Keyword arguments given to the constructor are stored in ``self.options``
    so that ``LoaderPipeline(default_loader_kwargs=..., per_suffix_kwargs=...)``
    can actually configure the concrete loaders.
    """

    def __init__(self, **options) -> None:
        self.options: dict = options

    @abstractmethod
    def load(self, path: Path | str) -> list[Document]:
        raise NotImplementedError
