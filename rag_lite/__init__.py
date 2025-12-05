"""
轻量化 RAG 原型包。

模块划分遵循 KISS/SOLID：配置、数据加载、向量库与主流水线互相解耦，
用户可在不修改核心逻辑的情况下替换模型或存储。
"""

from importlib import metadata

__all__ = ["__version__"]


def _load_version() -> str:
    try:
        return metadata.version("rag-lite")
    except metadata.PackageNotFoundError:
        return "0.1.0"


__version__ = _load_version()
