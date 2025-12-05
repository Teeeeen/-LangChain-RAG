from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from .config import RagSettings


def _resolve_ollama_model(model_name: str) -> tuple[bool, str]:
    normalized = model_name.strip()
    lowered = normalized.lower()
    if lowered.startswith("ollama:"):
        return True, normalized.split(":", 1)[1]
    if lowered.startswith("ollama/"):
        return True, normalized.split("/", 1)[1]
    if "/" not in normalized and ":" not in normalized and lowered.startswith("bge-"):
        return True, normalized
    return False, normalized


def create_embeddings(model_name: str, base_url: str) -> Embeddings:
    """Wrap embedding creation so callers stay decoupled from implementation (DIP)."""
    use_ollama, resolved_name = _resolve_ollama_model(model_name)
    if use_ollama:
        # Use Ollama embedding endpoint (default model tag when present).
        return OllamaEmbeddings(model=resolved_name, base_url=base_url)

    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    query_instruction = None

    # BGE-M3 需要 trust_remote_code 并推荐显式查询提示，以获得稳定的检索表现。
    if "bge-m3" in resolved_name.lower():
        model_kwargs["trust_remote_code"] = True
        query_instruction = "为这个句子生成表示以用于检索相关文章："

    return HuggingFaceBgeEmbeddings(
        model_name=resolved_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction=query_instruction,
    )


def build_vector_store(documents: Iterable[Document], settings: RagSettings) -> Chroma:
    persist_dir = Path(settings.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = create_embeddings(settings.embedding_model, settings.ollama_base_url)
    return Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=settings.collection_name,
    )


def load_vector_store(settings: RagSettings) -> Chroma:
    persist_dir = Path(settings.persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Missing vector store directory: {persist_dir}. Run ingest first."
        )
    embeddings = create_embeddings(settings.embedding_model, settings.ollama_base_url)
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=settings.collection_name,
        embedding_function=embeddings,
    )
