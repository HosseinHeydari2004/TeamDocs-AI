# rag/main_loader.py

import os


os.environ.setdefault("USER_AGENT", "TeamDocs-AI/1.0")  # must be set before WebBaseLoader is imported

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from devmind.rag.loaders import DocsLoader, PDFLoader, MarkdownLoader, TXTLoader, PPTXLoader
from devmind.rag.loaders.web_loader.web import WebLoader

from .code_loader import (
    SQLCodeLoader,
    CppCodeLoader,
    CSharpCodeLoader,
    GoCodeLoader,
    HTMLFileLoader,
    JavaCodeLoader,
    JavaScriptCodeLoader,
    KotlinCodeLoader,
    PythonCodeLoader,
    RustCodeLoader,
    TypeScriptCodeLoader,
)
from typing import Iterable, TypeAlias

SourceLike: TypeAlias = str | Path | Iterable[str | Path]


class LoaderPipeline:
    """
    Single entry point for loading documents and source code.

    ``load(source)`` auto-detects what it was given: a file, a directory, a ZIP
    archive, a URL (``http``/``https``), or a list/tuple/set of any of these.
    English documents only.

    The pipeline dispatches each source to an extension-specific loader
    (documents, web pages, or source code), normalizes metadata on every
    returned :class:`~langchain_core.documents.Document`, and handles ZIP
    archives recursively with protection against path-traversal (Zip Slip).

    Attributes
    ----------
    default_kwargs : dict
        Keyword arguments passed to every instantiated loader.
    per_suffix_kwargs : dict of str to dict
        Per-extension keyword arguments that override ``default_kwargs``.
        Keys are lower-cased file suffixes (e.g. ``".pptx"``).

    Examples
    --------
    >>> loader = LoaderPipeline()
    >>> docs = loader.load("report.pdf")
    >>> docs = loader.load("main.py")
    >>> docs = loader.load("data/")
    >>> docs = loader.load("archive.zip")
    >>> docs = loader.load("https://example.com/docs")
    >>> docs = loader.load(["a.pdf", "b.pptx", "main.py", "data/"])
    """

    _SUPPORTED: dict[str, type] = {
        # Documents
        ".pdf": PDFLoader,
        ".txt": TXTLoader,
        ".md": MarkdownLoader,
        ".doc": DocsLoader,
        ".docx": DocsLoader,
        ".docs": DocsLoader,
        ".pptx": PPTXLoader,
        # Code
        ".sql": SQLCodeLoader,
        ".cpp": CppCodeLoader,
        ".cc": CppCodeLoader,
        ".cxx": CppCodeLoader,
        ".hpp": CppCodeLoader,
        ".h": CppCodeLoader,
        ".cs": CSharpCodeLoader,
        ".go": GoCodeLoader,
        ".html": HTMLFileLoader,
        ".htm": HTMLFileLoader,
        ".java": JavaCodeLoader,
        ".js": JavaScriptCodeLoader,
        ".mjs": JavaScriptCodeLoader,
        ".cjs": JavaScriptCodeLoader,
        ".jsx": JavaScriptCodeLoader,
        ".kt": KotlinCodeLoader,
        ".kts": KotlinCodeLoader,
        ".py": PythonCodeLoader,
        ".rs": RustCodeLoader,
        ".ts": TypeScriptCodeLoader,
        ".tsx": TypeScriptCodeLoader,
    }
    """Mapping of lower-cased file suffix to the loader class that handles it."""

    _ARCHIVE_SUFFIXES: set[str] = {".zip"}
    """Archive suffixes extracted and loaded recursively."""

    _URL_PREFIXES: tuple[str, ...] = ("http://", "https://")
    """URL schemes routed to the web loader."""

    _MAX_ZIP_DEPTH: int = 3
    """Maximum nesting depth for recursively extracted ZIP archives."""

    def __init__(
            self,
            default_loader_kwargs: dict | None = None,
            per_suffix_kwargs: dict[str, dict] | None = None,
    ) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        default_loader_kwargs : dict, optional
            Keyword arguments shared by every loader
            (e.g. ``{"mode": "elements"}``). Defaults to ``None`` (empty).
        per_suffix_kwargs : dict of str to dict, optional
            Per-extension keyword arguments that override the shared ones
            (e.g. ``{".pptx": {"mode": "elements"}}``). Keys are normalized
            to lower case. Defaults to ``None`` (empty).

        Examples
        --------
        >>> loader = LoaderPipeline(
        ...     default_loader_kwargs={"mode": "elements"},
        ...     per_suffix_kwargs={".pptx": {"mode": "single"}},
        ... )
        """
        self.default_kwargs: dict = default_loader_kwargs or {}
        self.per_suffix_kwargs: dict[str, dict] = {
            s.lower(): kw for s, kw in (per_suffix_kwargs or {}).items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: SourceLike) -> list[Document]:
        """
        Load any supported source by auto-detecting its type.

        Dispatches to :meth:`load_many` for sequences, :meth:`load_url` for
        URLs, :meth:`load_directory` for directories, and :meth:`load_file`
        for files.

        Parameters
        ----------
        source : str, pathlib.Path, list, tuple, or set
            A single path/URL, or a collection of paths/URLs.

        Returns
        -------
        list of Document
            Loaded documents with normalized ``metadata["source"]`` and
            ``metadata["file_type"]`` fields.

        Examples
        --------
        >>> loader = LoaderPipeline()
        >>> loader.load("report.pdf")
        >>> loader.load("main.py")
        >>> loader.load("data/")
        >>> loader.load("archive.zip")
        >>> loader.load("https://example.com/docs")
        >>> loader.load(["a.pdf", "b.pptx", "main.py", "data/"])
        """
        if isinstance(source, (list, tuple, set)):
            return self.load_many(source)

        if self._is_url(source):
            return self.load_url(str(source))

        path = Path(source)
        if path.is_dir():
            return self.load_directory(path)

        return self.load_file(path)

    def get_loader(self, file_path: str | Path):
        """
        Return an instantiated loader for the given file's suffix.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path whose suffix determines which loader class is used.

        Returns
        -------
        object
            A loader instance constructed as ``LoaderClass(str(path), **kwargs)``.
            The concrete type depends on ``file_path``'s suffix.

        Raises
        ------
        ValueError
            If the file's suffix is not present in ``_SUPPORTED``.

        Examples
        --------
        >>> loader = LoaderPipeline()
        >>> pdf_loader = loader.get_loader("report.pdf")
        >>> sql_loader = loader.get_loader("schema.sql")
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        loader_cls = self._SUPPORTED.get(suffix)
        if loader_cls is None:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Allowed formats: {sorted(self._SUPPORTED.keys())}"
            )

        kwargs = {**self.default_kwargs, **self.per_suffix_kwargs.get(suffix, {})}
        return loader_cls()

    def load_file(self, file_path: str | Path) -> list[Document]:
        """
        Load a single file.

        ZIP archives are extracted into a temporary directory and loaded
        recursively; all other files are loaded with their suffix-specific
        loader.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the file to load. Must exist and be a regular file.

        Returns
        -------
        list of Document
            Loaded documents. ZIP members additionally carry
            ``metadata["archive_source"]`` and ``metadata["archive_member"]``.

        Raises
        ------
        FileNotFoundError
            If ``file_path`` does not exist.
        ValueError
            If ``file_path`` is not a file, or if a ZIP archive is corrupt.

        Examples
        --------
        >>> loader = LoaderPipeline()
        >>> docs = loader.load_file("report.pdf")
        >>> docs = loader.load_file("bundle.zip")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() in self._ARCHIVE_SUFFIXES:
            return self._load_zip(path, depth=0)

        return self._load_single_file(path)

    def load_url(self, url: str) -> list[Document]:
        """
        Load a web page with the internal :class:`WebLoader`.

        Parameters
        ----------
        url : str
            HTTP or HTTPS URL to fetch.

        Returns
        -------
        list of Document
            Scraped documents with ``metadata["source"]`` set to ``url`` and
            ``metadata["file_type"]`` set to ``"web"`` (unless already present).

        Examples
        --------
        >>> loader = LoaderPipeline()
        >>> docs = loader.load_url("https://example.com/docs")
        """
        docs = WebLoader().load(url=url)

        for doc in docs:
            doc.metadata.setdefault("source", url)
            doc.metadata.setdefault("file_type", "web")

        return docs

    def load_many(self, sources: Iterable[str | Path]) -> list[Document]:
        """
        Load multiple sources, tolerating individual failures.

        Each source is loaded via :meth:`load`. If a source raises, the error
        is printed to stdout and processing continues with the next source.

        Parameters
        ----------
        sources : iterable of str or pathlib.Path
            Sources to load (files, directories, archives, or URLs).

        Returns
        -------
        list of Document
            Concatenated documents from every source that loaded successfully.

        Examples
        --------
        >>> loader = LoaderPipeline()
        >>> docs = loader.load_many(["a.pdf", "bad.xyz", "b.pptx"])
        Skipped bad.xyz: Unsupported file format: '.xyz'. ...
        """
        all_docs: list[Document] = []

        for s in sources:
            try:
                all_docs.extend(self.load(s))
            except Exception as exc:
                print(f"Skipped {s}: {exc}")

        return all_docs

    def load_directory(
            self,
            directory: str | Path,
            recursive: bool = True,
    ) -> list[Document]:
        """
        Load every supported file (and archive) inside a directory.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory to scan. Must exist and be a directory.
        recursive : bool, default True
            If True, search subdirectories recursively (``**/*``); otherwise
            only the top level (``*``).

        Returns
        -------
        list of Document
            Loaded documents from all supported files found. Unsupported
            files are skipped silently; individual load failures are reported
            by :meth:`load_many`.

        Raises
        ------
        NotADirectoryError
            If ``directory`` is not a valid directory.

        Examples
        --------
        >>> loader = LoaderPipeline()
        >>> docs = loader.load_directory("data/")
        >>> docs = loader.load_directory("data/", recursive=False)
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Invalid directory: {directory}")

        allowed = set(self._SUPPORTED) | self._ARCHIVE_SUFFIXES
        pattern = "**/*" if recursive else "*"
        paths = [
            p for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in allowed
        ]
        return self.load_many(paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_url(self, source) -> bool:
        """
        Return whether ``source`` is an HTTP or HTTPS URL string.

        Parameters
        ----------
        source : object
            Candidate source; non-string values always return ``False``.

        Returns
        -------
        bool
            ``True`` if ``source`` is a string starting with a supported URL
            prefix, otherwise ``False``.
        """
        return isinstance(source, str) and source.lower().startswith(self._URL_PREFIXES)

    def _load_single_file(self, path: str | Path) -> list[Document]:
        """
        Load one supported file and normalize its metadata.

        Parameters
        ----------
        path : pathlib.Path
            Path to a file whose suffix is present in ``_SUPPORTED``.

        Returns
        -------
        list of Document
            Documents from the resolved loader with ``metadata["source"]``
            and ``metadata["file_type"]`` populated (if not already set).
        """
        docs = self.get_loader(file_path=path).load(path)

        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("file_type", path.suffix.lower().lstrip("."))

        return docs

    def _load_zip(self, zip_path: Path, depth: int) -> list[Document]:
        """
        Extract a ZIP archive to a temporary directory and load its entries.

        Supported files are loaded directly; nested archives are recursed up
        to ``_MAX_ZIP_DEPTH``. Entries whose resolved path escapes the
        extraction root are skipped (Zip Slip protection).

        Parameters
        ----------
        zip_path : pathlib.Path
            Path to the ZIP archive to extract.
        depth : int
            Current recursion depth. Must be ``<= _MAX_ZIP_DEPTH`` to proceed.

        Returns
        -------
        list of Document
            Documents from all supported entries. Each document carries
            ``metadata["archive_source"]`` (the archive path) and
            ``metadata["archive_member"]`` (the relative path inside it).
            Returns an empty list if ``depth`` exceeds ``_MAX_ZIP_DEPTH``.

        Raises
        ------
        ValueError
            If the archive is not a valid ZIP file.

        Notes
        -----
        The temporary extraction directory is always removed, even on error.
        """
        if depth > self._MAX_ZIP_DEPTH:
            return []

        all_docs: list[Document] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="mainloader_zip_"))

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                root = tmp_dir.resolve()
                for member in zf.infolist():
                    target = (tmp_dir / member.filename).resolve()
                    if not target.is_relative_to(root):
                        continue  # Zip Slip attempt
                    zf.extract(member, tmp_dir)

            for file in tmp_dir.rglob("*"):
                if not file.is_file():
                    continue

                suffix = file.suffix.lower()

                if suffix in self._ARCHIVE_SUFFIXES:
                    all_docs.extend(self._load_zip(file, depth + 1))
                elif suffix in self._SUPPORTED:
                    docs = self._load_single_file(file)
                    for doc in docs:
                        doc.metadata["archive_source"] = str(zip_path)
                        doc.metadata["archive_member"] = str(file.relative_to(tmp_dir))
                    all_docs.extend(docs)

        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP archive: {zip_path}") from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return all_docs

