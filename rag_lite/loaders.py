from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def load_documents(data_path: Path) -> List[Document]:
    """Load supported documents under the given directory."""
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在 {data_path}")

    documents: List[Document] = []
    for file_path in data_path.rglob("*"):
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
    if not documents:
        raise ValueError(f"{data_path} 中未找到可解析的文件")
    return documents


def split_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Apply a unified split strategy to avoid duplicated configs (DRY)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    return splitter.split_documents(list(documents))
