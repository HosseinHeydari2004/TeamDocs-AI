from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader as LCBSHTMLLoader
from devmind.rag.loaders.base import BaseLoader

class HTMLFileLoader(BaseLoader):
    EXTENSIONS = {".html", ".htm"}
    SKIP_DIRS = {".git", "env", "venv", ".venv", "node_modules", "build", "dist", ".idea"}

    def load(self, path):
        path = Path(path)
        files = [path] if path.is_file() else sorted(
            p for p in path.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS and not (self.SKIP_DIRS & set(p.parts))
        )

        docs = []
        for file in files:
            try:
                loader = LCBSHTMLLoader(str(file), open_encoding="utf-8")
                for doc in loader.load():
                    doc.metadata["language"] = "html"
                    doc.metadata["file_name"] = file.name
                    docs.append(doc)
            except Exception as e:
                print(f"skip {file.name}: {e}")
        return docs
