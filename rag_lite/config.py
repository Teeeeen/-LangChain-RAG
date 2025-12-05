from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


@dataclass(frozen=True)
class RagSettings:
    """Central place for RAG parameters to keep defaults consistent (SRP)."""

    data_path: Path = Path("data/sample_docs")
    persist_dir: Path = Path("data/vector_store")
    collection_name: str = "local_rag"
    embedding_model: str = "ollama:bge-m3"
    llm_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 80

    def with_override(self, **kwargs: Any) -> "RagSettings":
        """Return a new instance without mutating the current one (OCP/DIP)."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": str(self.data_path),
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "ollama_base_url": self.ollama_base_url,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


def load_settings(**overrides: Any) -> RagSettings:
    load_dotenv()
    base = RagSettings(
        data_path=Path(os.getenv("RAG_DATA_PATH", RagSettings.data_path)),
        persist_dir=Path(os.getenv("RAG_PERSIST_DIR", RagSettings.persist_dir)),
        collection_name=os.getenv("RAG_COLLECTION", RagSettings.collection_name),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", RagSettings.embedding_model),
        llm_model=os.getenv("RAG_LLM_MODEL", RagSettings.llm_model),
        ollama_base_url=os.getenv(
            "RAG_OLLAMA_BASE_URL", RagSettings.ollama_base_url
        ),
        temperature=float(os.getenv("RAG_TEMPERATURE", RagSettings.temperature)),
        top_k=int(os.getenv("RAG_TOP_K", RagSettings.top_k)),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", RagSettings.chunk_size)),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", RagSettings.chunk_overlap)),
    )
    return base.with_override(**overrides)
