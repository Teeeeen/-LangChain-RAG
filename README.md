# 轻量化本地 RAG 系统

基于 LangChain + Chroma + 本地 LLM（默认 Ollama）的一站式 RAG 原型，支持 CLI 问答、FastAPI 接口和 Streamlit 前端。默认向量库目录在 `data/`，前端已简化为单文件 Streamlit（`frontend/app.py`）。

## 环境要求
- Python 3.10+
- 已安装 [Ollama](https://ollama.com/) 并可运行；需要拉取 `qwen3:8b` 与 `bge-m3`（用于 embedding）

## 快速开始
1) 创建虚拟环境并安装依赖
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2) 准备本地模型（Ollama）
   ```powershell
   ollama pull qwen3:8b
   ollama pull bge-m3
   ollama serve   # 或 ollama run qwen3:8b，退出用 /exit
   ```
   - 如在 WSL/远程主机运行 Ollama，可设置 `RAG_OLLAMA_BASE_URL` 指向服务地址（默认 `http://localhost:11434`）
   - 若希望在 WSL2/本地使用 GPU 运行 HuggingFace Embedding，可设置 `RAG_DEVICE=cuda`（默认自动检测 CUDA，失败则回退 CPU）

3) 构建向量库（默认使用 Ollama BGE-M3 embedding）
   ```powershell
   python -m rag_lite.cli ingest `
     --data-path data/sample_docs `
     --persist-dir data/vector_store `
     --embedding ollama:bge-m3
   ```
   - 首次会通过 Ollama 调用 `bge-m3` 生成向量写入 `data/vector_store/`
   - 若有自定义语料，将文件放入新目录并替换 `--data-path`

4) CLI 问答
   ```powershell
   python -m rag_lite.cli ask `
     --question "轻量化 RAG 的核心步骤是什么？" `
     --persist-dir data/vector_store `
     --embedding ollama:bge-m3 `
     --model qwen3:8b
   ```

5) 启动 REST API（供外部/Streamlit 调用）
   ```powershell
   python -m rag_lite.server
   # 或 uvicorn rag_lite.server:app --host 0.0.0.0 --port 8000 --reload
   ```
   常用接口：`GET /api/health`、`GET /api/config`、`POST /api/chat`

6) 运行 Streamlit 前端
   ```powershell
   streamlit run frontend/app.py --server.port 8501
   ```
   在浏览器打开 http://localhost:8501，侧边栏可调整模型、向量库路径并触发重新索引；主区域提供聊天窗口和来源/耗时展示。

7) 性能测试脚本
   ```powershell
   python scripts/perf_report.py `
     --question "RAG 管线的瓶颈是什么？" `
     --runs 10 `
     --persist-dir data/vector_store `
     --embedding ollama:bge-m3 `
     --model qwen3:8b
   ```
   输出检索/生成耗时、总耗时、峰值内存与示例答案，可记录到 `docs/performance_report.md`。

## 工程模块与流程
- 核心模块
  - `rag_lite/config.py`：集中默认配置与环境变量加载。
  - `rag_lite/loaders.py`：文档加载与分块（文本/PDF + 递归切分）。
  - `rag_lite/vectorstore.py`：Embedding 工厂（Ollama/HF）与 Chroma 构建/加载。
  - `rag_lite/pipeline.py`：统一的 RAG 流水线，封装 ingest/ask。
  - `rag_lite/cli.py`：Typer CLI（`ingest`/`ask`）。
  - `rag_lite/server.py`：FastAPI 接口（/api/health、/api/chat 等）。
  - `frontend/app.py`：Streamlit 前端（侧边栏配置 + 聊天主区）。
  - `scripts/perf_report.py`：性能与 RAGAS 质量评测脚本。

- 典型流程
  1) Ingest：`load_documents` -> `split_documents` -> `create_embeddings` -> `Chroma.from_documents` 持久化。
  2) Ask：`load_vector_store` -> `retriever.get_relevant_documents` -> 模板渲染 -> `Ollama` 生成 -> 返回答案/来源/耗时。
  3) 前后端：CLI/Server/Streamlit 均复用同一 `RagPipeline`，配置可通过参数或环境变量注入，确保 DRY/OCP。

## 配置项
- 环境变量可覆盖默认值：`RAG_DATA_PATH`、`RAG_PERSIST_DIR`、`RAG_COLLECTION`、`RAG_EMBEDDING_MODEL`、`RAG_LLM_MODEL`、`RAG_OLLAMA_BASE_URL`、`RAG_TEMPERATURE`、`RAG_TOP_K`、`RAG_CHUNK_SIZE`、`RAG_CHUNK_OVERLAP`
- CLI 选项优先级最高，会覆盖环境变量和默认配置

## 常见问题
- `ModuleNotFoundError: rag_lite`：确保当前路径在项目根目录，或执行 `pip install -e .`
- 未找到向量库：先运行 `python -m rag_lite.cli ingest ...` 生成 `data/vector_store/`
- Ollama 相关报错：确认 `ollama serve` 正在运行且已拉取 `qwen3:8b`；如需更换模型，用 `--model`/`--embedding` 传入新名称
