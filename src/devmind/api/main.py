import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from devmind.services.qa_service import QAService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TeamDocs-AI",
    description="Repository QA using RAG and GitHub MCP",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    owner: str = "HosseinHeydari2004"
    repo: str = "TeamDocs-AI"
    openrouter_api_key: str = Field(min_length=1)
    github_token: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        service = QAService(
            openrouter_api_key=request.openrouter_api_key,
            github_token=request.github_token,
        )
        return await service.answer(
            question=request.question,
            owner=request.owner,
            repo=request.repo,
        )
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Repository QA request failed.",
        )