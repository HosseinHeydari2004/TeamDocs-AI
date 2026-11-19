"""Pydantic request/response models (the public API contract; see /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Natural-language question.")
    top_k: int | None = Field(
        None, ge=1, le=20, description="Passages to retrieve (default: server setting)."
    )
    mode: Literal["hybrid", "dense", "bm25"] | None = Field(None, description="Retrieval strategy.")
    filename: str | None = Field(None, max_length=255, description="Restrict retrieval to one document.")
    generate: bool = Field(True, description="false = return retrieved sources only (no LLM call).")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "How does the retry policy work?",
                "top_k": 5,
                "mode": "hybrid",
                "generate": True,
            }
        }
    }


class SourceOut(BaseModel):
    index: int = Field(..., description="The [n] marker used in the answer text.")
    filename: str
    page: int | None = None
    section: str | None = None
    source_url: str | None = Field(None, description="Original URL when the passage comes from a web page.")
    excerpt: str
    cited: bool = Field(..., description="True when the answer text cites this passage.")
    score: float = Field(..., description="Ranking score of the retrieval method (RRF score in hybrid mode).")
    dense_score: float | None = Field(None, description="Cosine similarity of the passage to the query.")
    bm25_score: float | None = Field(None, description="Raw BM25 score (corpus dependent).")
    doc_id: str | None = None
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str | None
    sources: list[SourceOut]
    insufficient_evidence: bool
    generated: bool
    mode: str
    model: str | None = None
    timings_ms: dict[str, float]


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    size_bytes: int | None = None
    ingested_at: str | None = None
    pages: int | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total_documents: int
    total_chunks: int


class IngestResponse(BaseModel):
    status: Literal["indexed", "replaced", "unchanged", "duplicate"]
    doc_id: str
    filename: str
    chunk_count: int
    replaced_chunks: int
    message: str
    duration_ms: float


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool | str | None]
    documents: int | None = None
    chunks: int | None = None
    startup_ms: dict[str, float] = {}


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------- Main Agent chat
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The user's message.")
    history: list[ChatTurn] = Field(default_factory=list, max_length=20, description="Previous turns (the server is stateless).")

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()

    model_config = {"json_schema_extra": {"example": {"message": "How does the retry policy work, and which open issues mention it in octocat/Hello-World?"}}}


class AgentErrorOut(BaseModel):
    code: str
    message: str


class AgentRunOut(BaseModel):
    agent: Literal["rag", "github"]
    ok: bool
    tool_calls: list[str]
    model: dict[str, Any] | None = None
    duration_ms: float
    insufficient_evidence: bool = False
    error: AgentErrorOut | None = None


class GitHubRefOut(BaseModel):
    kind: str
    repository: str | None = None
    label: str
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    agents_used: list[AgentRunOut] = Field(description="Which sub-agents the Main Agent called (empty = it answered directly or asked a question).")
    sources: list[SourceOut]
    github: list[GitHubRefOut]
    warnings: list[str]
    model: dict[str, Any]
    duration_ms: float


# ---------------------------------------------------------------- URL import / settings
class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, description="http(s) page to download and index.")


class LLMSelection(BaseModel):
    provider: Literal["auto", "openrouter", "ollama"] | None = None
    openrouter_model: str | None = Field(None, max_length=200)
    ollama_model: str | None = Field(None, max_length=200)


class StatusResponse(BaseModel):
    ready: bool
    checked: bool
    llm: dict[str, Any]
    github: dict[str, Any]
    documents: int | None = None
    chunks: int | None = None
