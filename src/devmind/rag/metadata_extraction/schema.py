from pydantic import BaseModel, Field
from typing import Literal

class SemanticMetadata(BaseModel):
    title: str | None = None
    summary: str | None = None
    document_type: Literal[
        "documentation",
        "tutorial",
        "api_reference",
        "code",
        "readme",
        "article",
        "report",
        "unknown",
    ] = "unknown"
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    language: str | None = None
    programming_language: str | None = None