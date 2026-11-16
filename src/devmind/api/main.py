
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from devmind.services.qa_service import QAService


app = FastAPI(
    title="TeamDocs-AI",
    description="Repository QA using RAG and GitHub MCP",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    owner: str = "HosseinHeydari2004"
    repo: str = "TeamDocs-AI"


@lru_cache
def get_service():
    return QAService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        service = get_service()

        return await service.answer(
            question=request.question,
            owner=request.owner,
            repo=request.repo,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
