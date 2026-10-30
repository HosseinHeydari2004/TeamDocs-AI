from pathlib import Path

from langchain_community.document_loaders import TextLoader as LCTextLoader
from devmind.rag.loaders.base import BaseLoader

class SQLCodeLoader(BaseLoader):
    EXTENSIONS = {".sql", ".ddl", ".psql", ".pgsql", ".mysql"}
    SKIP_DIRS = {".git", "env", "venv", ".venv", "node_modules", "build", "dist", ".idea"}

    def __init__(self, max_file_size=5_000_000):
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
                print(f"skip {file.name}: file too large")
                continue
            try:
                loader = LCTextLoader(str(file), encoding="utf-8", autodetect_encoding=True)
                for doc in loader.load():
                    doc.metadata["language"] = "sql"
                    docs.append(doc)
            except Exception as e:
                print(f"skip {file.name}: {e}")
        return docs

