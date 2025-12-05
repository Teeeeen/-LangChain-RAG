from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import RagSettings, load_settings
from .pipeline import RagPipeline


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_items=1)
    persist_dir: Optional[str] = None
    embedding: Optional[str] = None
    model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    temperature: Optional[float] = Field(default=None, ge=0, le=1)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    metrics: Dict[str, float]


def _build_pipeline(overrides: Dict[str, object] | None = None) -> RagPipeline:
    overrides = overrides or {}
    normalized: Dict[str, object] = {}
    if overrides.get("persist_dir"):
        normalized["persist_dir"] = Path(overrides["persist_dir"])
    if overrides.get("embedding_model"):
        normalized["embedding_model"] = overrides["embedding_model"]
    if overrides.get("llm_model"):
        normalized["llm_model"] = overrides["llm_model"]
    if overrides.get("ollama_base_url"):
        normalized["ollama_base_url"] = overrides["ollama_base_url"]
    if overrides.get("top_k") is not None:
        normalized["top_k"] = overrides["top_k"]
    if overrides.get("temperature") is not None:
        normalized["temperature"] = overrides["temperature"]
    settings = load_settings(**normalized)
    return RagPipeline(settings)


def _default_settings() -> RagSettings:
    return load_settings()


def create_app() -> FastAPI:
    """FastAPI app factory reused by server/CLI (DRY)."""

    app = FastAPI(title="RAG Lite API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, object]:
        settings = _default_settings()
        return {
            "status": "ok",
            "model": settings.llm_model,
            "embedding": settings.embedding_model,
        }

    @app.get("/api/config")
    def get_config() -> Dict[str, object]:
        return _default_settings().to_dict()

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        question = next(
            (m.content.strip() for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        if not question:
            raise HTTPException(status_code=400, detail="未找到用户问题")

        pipeline = _build_pipeline(
            {
                "persist_dir": request.persist_dir,
                "embedding_model": request.embedding,
                "llm_model": request.model,
                "ollama_base_url": request.ollama_base_url,
                "top_k": request.top_k,
                "temperature": request.temperature,
            }
        )
        result = pipeline.ask(question)
        return ChatResponse(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            metrics=result["metrics"],
        )

    return app


app = create_app()


def main():
    uvicorn.run(
        "rag_lite.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
