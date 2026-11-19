import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader as LCTextLoader

from rag.loaders.base import BaseLoader

logger = logging.getLogger(__name__)

class SQLCodeLoader(BaseLoader):
    EXTENSIONS = {".sql", ".ddl", ".psql", ".pgsql", ".mysql"}
    SKIP_DIRS = {".git", "env", "venv", ".venv", "node_modules", "build", "dist", ".idea"}

    def __init__(self, max_file_size: int = 5_000_000, **options):
        super().__init__(**options)
        self.max_file_size = max_file_size

    def load(self, path):
        path = Path(path)
        files = [path] if path.is_file() else sorted(
            p for p in path.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS and not (self.SKIP_DIRS & set(p.parts))
        )

        docs = []
        for file in files:
            if file.stat().st_size > self.max_file_size:
                logger.warning("skip %s: file too large", file.name)
                continue
            try:
                loader = LCTextLoader(str(file), encoding="utf-8", autodetect_encoding=True)
                for doc in loader.load():
                    doc.metadata["language"] = "sql"
                    docs.append(doc)
            except Exception as e:
                logger.warning("skip %s: %s", file.name, e)
        return docs
