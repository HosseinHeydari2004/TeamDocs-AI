from pathlib import Path

from langchain_core.documents import Document

from rag.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """Load a PDF.

    ``backend`` option:
      * ``"pypdf"`` (default): one Document per page, no model downloads, fast.
        ``metadata["page"]`` is the zero-based page index.
      * ``"unstructured"``: layout-aware element extraction
        (``strategy`` option, default ``"hi_res"``). Needs the
        ``unstructured[pdf]`` extras, poppler and tesseract, and downloads
        layout models on first use.
    """

    def load(self, path: Path | str) -> list[Document]:
        backend = self.options.get("backend", "pypdf")

        if backend == "pypdf":
            from langchain_community.document_loaders import PyPDFLoader

            return PyPDFLoader(str(path)).load()

        if backend == "unstructured":
            from langchain_community.document_loaders import UnstructuredPDFLoader

            return UnstructuredPDFLoader(
                file_path=str(path),
                mode="elements",
                strategy=self.options.get("strategy", "hi_res"),
                languages=["eng"],
                infer_table_structure=True,
            ).load()

        raise ValueError(f"Unknown PDF backend: {backend!r}")
