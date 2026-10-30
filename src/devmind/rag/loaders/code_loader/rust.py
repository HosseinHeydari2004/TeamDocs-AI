from pathlib import Path

from langchain_community.document_loaders import TextLoader as LCTextLoader


class RustCodeLoader:
    EXTENSIONS = {".rs"}
    SKIP_DIRS = {".git", "env", "venv", ".venv", "target", ".idea"}

    def load(self, path):
        path = Path(path)
        files = [path] if path.is_file() else sorted(
            p for p in path.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS and not (self.SKIP_DIRS & set(p.parts))
        )

        docs = []
        for file in files:
            try:
                loader = LCTextLoader(str(file), encoding="utf-8", autodetect_encoding=True)
                for doc in loader.load():
                    doc.metadata["language"] = "rust"
                    docs.append(doc)
            except Exception as e:
                print(f"skip {file.name}: {e}")
        return docs
