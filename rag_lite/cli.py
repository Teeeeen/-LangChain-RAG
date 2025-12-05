from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import RagSettings, load_settings
from .pipeline import RagPipeline

app = typer.Typer(help="轻量化本地 RAG CLI")


def _build_settings(
    data_path: Optional[Path],
    persist_dir: Optional[Path],
    embedding: Optional[str],
    model: Optional[str],
    ollama_base_url: Optional[str],
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    top_k: Optional[int],
    temperature: Optional[float],
) -> RagSettings:
    return load_settings(
        data_path=data_path,
        persist_dir=persist_dir,
        embedding_model=embedding,
        llm_model=model,
        ollama_base_url=ollama_base_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        temperature=temperature,
    )


@app.command()
def ingest(
    data_path: Path = typer.Option(Path("data/sample_docs"), help="原始文档目录"),
    persist_dir: Path = typer.Option(Path("data/vector_store"), help="向量库存储目录"),
    embedding: str = typer.Option("ollama:bge-m3", help="Embedding 模型名称"),
    ollama_base_url: str = typer.Option(
        None, help="Ollama 服务地址（默认 http://localhost:11434）"
    ),
    chunk_size: int = typer.Option(500, min=200, help="分块大小"),
    chunk_overlap: int = typer.Option(80, min=0, help="分块重叠"),
):
    """构建或刷新向量库。"""
    settings = _build_settings(
        data_path=data_path,
        persist_dir=persist_dir,
        embedding=embedding,
        model=None,
        ollama_base_url=ollama_base_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=None,
        temperature=None,
    )
    pipeline = RagPipeline(settings)
    stats = pipeline.ingest()
    typer.echo(
        f"[ingest] 加载 {stats['documents']} 个文档，生成 {stats['chunks']} 个分块，已写入 {persist_dir}"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="提问内容"),
    persist_dir: Path = typer.Option(Path("data/vector_store"), help="向量库存储目录"),
    embedding: str = typer.Option("ollama:bge-m3", help="Embedding 模型名称"),
    model: str = typer.Option("qwen3:8b", help="Ollama 模型名称"),
    ollama_base_url: str = typer.Option(
        None, help="Ollama 服务地址（默认 http://localhost:11434）"
    ),
    top_k: int = typer.Option(4, min=1, max=10, help="检索返回的文档数"),
    temperature: float = typer.Option(0.1, min=0.0, max=1.0, help="LLM 采样温度"),
):
    """针对知识库进行问答。"""
    settings = _build_settings(
        data_path=None,
        persist_dir=persist_dir,
        embedding=embedding,
        model=model,
        ollama_base_url=ollama_base_url,
        chunk_size=None,
        chunk_overlap=None,
        top_k=top_k,
        temperature=temperature,
    )
    pipeline = RagPipeline(settings)
    result = pipeline.ask(question)
    typer.echo(f"\nQ: {question}\nA: {result['answer']}\n")
    metrics = result.get("metrics", {})
    if metrics:
        typer.echo(
            f"retrieval: {metrics['retrieval_ms']:.1f} ms | "
            f"generation: {metrics['generation_ms']:.1f} ms"
        )
    typer.echo("Sources:")
    for idx, source in enumerate(result["sources"], start=1):
        typer.echo(f"  {idx}. {source}")


def main():
    app()


if __name__ == "__main__":
    main()
