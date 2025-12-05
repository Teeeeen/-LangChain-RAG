from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

import streamlit as st

# Ensure project root is importable when running `streamlit run frontend/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_lite.config import load_settings
from rag_lite.pipeline import RagPipeline

st.set_page_config(page_title="RAG Lite (Streamlit)", layout="wide")

DEFAULTS = load_settings()


@st.cache_resource(show_spinner=False)
def get_pipeline(settings_items: tuple) -> RagPipeline:
    settings: Dict[str, object] = dict(settings_items)
    normalized = settings.copy()
    normalized["data_path"] = Path(normalized["data_path"])
    normalized["persist_dir"] = Path(normalized["persist_dir"])
    rag_settings = load_settings(**normalized)
    return RagPipeline(rag_settings)


def sidebar_controls() -> tuple[Dict[str, object], bool, bool]:
    st.sidebar.header("配置")
    st.sidebar.caption("前端已切换 Streamlit · 推荐 qwen3:8b + BGE-M3（默认）")
    ollama_base_url = st.sidebar.text_input(
        "Ollama 地址", value=str(DEFAULTS.ollama_base_url)
    )
    data_path = st.sidebar.text_input("知识库目录", value=str(DEFAULTS.data_path))
    persist_dir = st.sidebar.text_input("向量库目录", value=str(DEFAULTS.persist_dir))
    embedding_model = st.sidebar.text_input("Embedding 模型", value=DEFAULTS.embedding_model)
    llm_model = st.sidebar.text_input("LLM 模型", value=DEFAULTS.llm_model)
    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=10, value=DEFAULTS.top_k)
    temperature = st.sidebar.slider(
        "温度", min_value=0.0, max_value=1.0, value=float(DEFAULTS.temperature), step=0.05
    )
    chunk_size = st.sidebar.number_input(
        "分块大小", min_value=200, max_value=2000, value=DEFAULTS.chunk_size, step=50
    )
    chunk_overlap = st.sidebar.number_input(
        "分块重叠", min_value=0, max_value=400, value=DEFAULTS.chunk_overlap, step=10
    )
    rebuild = st.sidebar.button("重新构建向量库", type="primary")
    clear = st.sidebar.button("清空对话")

    settings = {
        "data_path": data_path.strip() or str(DEFAULTS.data_path),
        "persist_dir": persist_dir.strip() or str(DEFAULTS.persist_dir),
        "embedding_model": embedding_model.strip() or DEFAULTS.embedding_model,
        "llm_model": llm_model.strip() or DEFAULTS.llm_model,
        "ollama_base_url": ollama_base_url.strip() or DEFAULTS.ollama_base_url,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
    }
    return settings, rebuild, clear


def render_messages(messages: List[Dict[str, object]]) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta", {})
            sources = meta.get("sources") or []
            metrics = meta.get("metrics") or {}
            if sources:
                st.caption("来源: " + " | ".join(sources))
            if metrics:
                st.caption(
                    f"retrieval {metrics.get('retrieval_ms', 0):.1f} ms · "
                    f"generation {metrics.get('generation_ms', 0):.1f} ms"
                )


def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def handle_ingest(pipeline: RagPipeline) -> None:
    with st.spinner("正在重建向量库..."):
        stats = pipeline.ingest()
    st.success(
        f"完成：{stats['documents']} 个文档，{stats['chunks']} 个分块，存储于 {pipeline.settings.persist_dir}"
    )


def handle_question(pipeline: RagPipeline, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("检索与生成中..."):
        result = pipeline.ask(question)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "sources": result.get("sources", []),
                "metrics": result.get("metrics", {}),
            },
        }
    )


def main():
    settings, rebuild, clear = sidebar_controls()
    settings_key = tuple(sorted(settings.items()))
    pipeline = get_pipeline(settings_key)

    ensure_session_state()
    if clear:
        st.session_state.messages = []

    st.title("RAG Lite · Streamlit")
    st.caption(
        f"LLM: {pipeline.settings.llm_model} | Embedding: {pipeline.settings.embedding_model} | TopK: {pipeline.settings.top_k}"
    )
    st.caption(
        f"Data: {pipeline.settings.data_path} | Persist: {pipeline.settings.persist_dir} | Chunk: {pipeline.settings.chunk_size}/{pipeline.settings.chunk_overlap}"
    )
    st.caption(
        f"Ollama: {pipeline.settings.ollama_base_url}"
    )

    if rebuild:
        handle_ingest(pipeline)

    render_messages(st.session_state.messages)

    prompt = st.chat_input("输入问题...")
    if prompt:
        try:
            handle_question(pipeline, prompt)
            st.rerun()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"请求失败: {exc}")


if __name__ == "__main__":
    main()
