"""Structured input/output contracts shared by the sub-agents, the Main Agent and the API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """A retrieved passage the answer may cite as ``[index]``."""

    index: int
    filename: str
    page: int | None = None
    section: str | None = None
    source_url: str | None = None
    excerpt: str
    cited: bool = False
    score: float = 0.0
    dense_score: float | None = None
    bm25_score: float | None = None
    doc_id: str | None = None
    chunk_id: str = ""


class GitHubRef(BaseModel):
    """A GitHub object the sub-agent actually read (shown next to the answer)."""

    kind: Literal["file", "directory", "issue", "pull_request", "commits", "code_search", "repositories"]
    repository: str | None = None
    label: str
    url: str | None = None


class SubAgentError(BaseModel):
    code: str
    message: str


class SubAgentResult(BaseModel):
    agent: Literal["rag", "github"]
    ok: bool
    answer: str | None = None
    insufficient_evidence: bool = False
    sources: list[SourceRef] = Field(default_factory=list)
    github: list[GitHubRef] = Field(default_factory=list)
    error: SubAgentError | None = None
    model: dict[str, Any] | None = None
    tool_calls: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0

    def for_supervisor(self) -> dict[str, Any]:
        """Compact JSON handed back to the Main Agent's LLM (never contains secrets)."""
        if not self.ok:
            return {"agent": self.agent, "ok": False, "error": self.error.model_dump() if self.error else None}
        payload: dict[str, Any] = {"agent": self.agent, "ok": True, "answer": self.answer}
        if self.agent == "rag":
            payload["insufficient_evidence"] = self.insufficient_evidence
            payload["sources"] = [
                {"n": s.index, "file": s.filename, **({"page": s.page} if s.page else {})} for s in self.sources if s.cited
            ] or [{"n": s.index, "file": s.filename} for s in self.sources[:5]]
        else:
            payload["github_objects"] = [g.label for g in self.github]
        return payload
