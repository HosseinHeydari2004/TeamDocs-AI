# Audit report

Scope: the repository as received (`TeamDocs-AI.rar`, 132 files). Findings are from reading the code, running
static checks (`ruff`), resolving dependencies against PyPI, and reproducing bugs with tests. Status shows what
was changed in this delivery. Items marked *by inspection* were not reproduced at runtime.

## Critical

| # | Finding | Status |
|---|---|---|
| C1 | `PPTXLoader.load()` built a loader and returned `None`; any `.pptx` crashed ingestion. | Fixed, regression test |
| C2 | Three import roots mixed (`src.rag…`, `rag…`, `devmind…`), a stray `src/__init__.py`, a stale `*.egg-info` describing a different layout and no `pyproject.toml`. Worked only for specific working directories. | Fixed: `src` layout, `pyproject.toml`, editable install |
| C3 | The only HTTP endpoint (`/chat`) built a new `QAService` per request, and `QAService` re-created its `rag_pipelines` cache in `__init__`. By inspection, every request constructed a new `RagPipeline`, i.e. a new `HuggingFaceEmbeddings` model, Chroma client and agent. | Fixed in the new API (resources built once in the FastAPI lifespan) |
| C4 | Semantic metadata extraction ran on *loaded elements* before chunking: one LLM call per paragraph for element-mode PDFs. `extract_by_source` existed but was unused. | Fixed: once per source, after chunking, off by default |

## High

* **Undeclared / incompatible dependencies**: `mcp` (imported), `python-pptx` and `chardet` (needed at runtime by the
  loaders) were missing. `chardet` 7.x crashes `langchain-community`'s encoding detector (`mime_type` kwarg) —
  reproduced; pinned to 5.2.0.
* `CleanerPipeline`: web documents raised `ValueError` before reaching the web cleaner; `.pptx` had no cleaner; the
  error path raised `AttributeError` (`document.metadata.type`).
* `document_type` collision: LLM-extracted `document_type` (e.g. `"code"`) selected the chunker, then was overwritten
  with the chunker name.
* `LoaderPipeline.get_loader` computed per-suffix kwargs and discarded them.
* BM25: returned top-k even with zero query-term overlap (noise fused into hybrid results); a filtered query rebuilt a
  BM25 index per request; the shared retriever's `k` was mutated per call (not thread-safe); no locking around the
  index rebuild.
* Chunk id = `sha256(source + text)`: edited files left stale chunks forever, identical paragraphs were silently
  dropped, and there was no update strategy (only `delete_by_source`).
* No structured citations: sources existed only as free text produced by the LLM; loader metadata contained absolute
  server paths (`source`, `file_directory`).
* `/chat` accepted API keys and a GitHub token in the JSON body and defaulted to a hard-coded repository.
* PDF default `strategy="hi_res"` (layout models downloaded at first use, slow).
* Chroma rejects nested/list metadata types; Unstructured and the metadata extractor both produce them.

## Medium / Low

`print`-based silent skips in 12 loaders; 25 lines of commented-out code in `rag_pipeline.py`; duplicated
assignments in `QAService.__init__`; Markdown chunker ignored the configured chunk size; no request timeouts on the
LLM client (OpenRouter's `timeout` is in **milliseconds**); no input validation; no tests; no Docker/CI; README
described no run instructions; `.env`-style config read at import time in several modules. Typos in module names
(`pipelline.py`, `preprossing/`) were left untouched to avoid breaking imports.

## Dependency audit

The original 109 pins resolve on PyPI without conflicts (checked with `pip install --dry-run`), but:

* **Missing**: `mcp`, `python-pptx`, `chardet`, `unstructured` extras (`docx`, `pptx`).
* **Unused** (no import anywhere, directly or via the used LangChain integrations): `faiss-cpu`, `streamlit`,
  `torchvision`, `timm`, `accelerate`, `onnx`, `spacy`, `nltk`(transitive only), `scikit-learn`, `scikit-network`,
  `scipy`, `pandas`(eval only), `pyarrow`, `datasets`, `numexpr`, `Bottleneck`, `google-cloud-*` (4),
  `langchain-google-genai`, `langchain-google-vertexai`, `google-genai`, `langchain-ollama`, `ollama`,
  `langchain-openai`, `openai`, `openrouter`, `langchain-experimental`, `langchain-classic`, `langgraph-*`(transitive),
  `opencv`-class and OCR tooling, `pi_heif`, `emoji`, `pyfiglet`, `tabulate`, `prettytable`, `rich`, `typer`, and
  pinned *transitive* packages (`pip`, `setuptools`, `wheel`, `starlette`, `pydantic_core`, …).
* `deepeval`/`pandas` are only used by `evaluation/` → moved to `requirements-optional.txt`.
* The new `requirements.txt` lists 20 direct dependencies; transitive versions are resolved by pip.


---

## Update: agent architecture, provider policy and URL import (Nov 2026)

| # | Finding (verified by reproduction or test) | Fix |
|---|---|---|
| 1 | OpenRouter key "not set" although configured: `.env` was resolved from the *working directory*; an empty `OPENROUTER_API_KEY` in the environment shadowed `.env`; a BOM broke the first variable; `devmind.config` loaded `.env` separately (import-time snapshot) and disagreed with the API's settings; no reload without restart | Single `Settings`: project-anchored `.env` lookup, blank env ignored, BOM/quotes/`Bearer`/placeholders cleaned, `key_source` diagnostics, `POST /api/v1/settings/reload`; `devmind.config` is a lazy view |
| 2 | `mcp` 2.x renamed `isError`→`is_error`: GitHub errors were silently treated as file content | `result_is_error()` compat helper |
| 3 | Repository download failed for every file in a sub-directory (parent dir never created) | `mkdir(parents=True)` |
| 4 | File reads from the real GitHub MCP server include a status line that was written into the file | `result_text()` prefers the embedded resource |
| 5 | A single add of >5,461 chunks crashed Chroma | Batched writes |
| 6 | `/ready` rescanned all chunk metadata on every poll (181 ms at 6k chunks) | Write-invalidated cache (0.6 ms) |
| 7 | Parallel backend (`devmind.api.main` with its own config/RAG) | Replaced by a shim onto the single app; `QAService` marked legacy |
| 8 | Web loader had no SSRF protection, swallowed all failures and was not reachable from API/UI | `SafeSession`, typed errors, `POST /api/v1/documents/url`, UI form |
| 9 | No agent layer, no local-model option | Main Agent (`create_deep_agent`), RAG + GitHub sub-agents, `ModelManager` (OpenRouter/Ollama policy, controlled fallback, real status checks), Settings UI |
