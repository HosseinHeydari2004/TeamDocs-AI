from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader as LCTextLoader
from devmind.rag.loaders.base import BaseLoader

class CppCodeLoader(BaseLoader):
    EXTENSIONS = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)

        if path.is_dir():
            files = sorted(p for p in path.rglob("*")
                           if p.suffix.lower() in self.EXTENSIONS)
        else:
            files = [path]

        docs = []
        for file in files:
            try:
                loader = LCTextLoader(str(file), encoding="utf-8",
                                      autodetect_encoding=True)
                for doc in loader.load():
                    doc.metadata["language"] = (
                        "c" if file.suffix.lower() in {".c", ".h"} else "cpp"
                    )
                    docs.append(doc)
            except Exception as e:
                print(f"skip {file.name}: {e}")

        return docs

