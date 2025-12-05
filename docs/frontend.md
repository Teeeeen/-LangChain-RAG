# Streamlit 前端说明

当前唯一维护的前端是 `frontend/app.py`（Streamlit）。为保持 KISS/YAGNI，已移除旧的静态 HTML/JS/CSS。

## 启动
```powershell
python -m rag_lite.server                      # 启动 FastAPI 后端
streamlit run frontend/app.py --server.port 8501
```
浏览器访问 `http://localhost:8501` 进行本地 RAG 问答。

## 功能
- 侧边栏：配置数据目录/向量库目录、Embedding/LLM、Ollama 地址、TopK、温度、分块大小/重叠；支持一键重建向量库与清空对话。
- 主区：聊天窗口，展示回答、来源列表、检索/生成耗时；支持多轮对话。
- 缓存：`st.cache_resource` 按配置复用 `RagPipeline`，避免重复加载。

## 推荐默认
- LLM：`qwen3:8b`（Ollama）
- Embedding：`ollama:bge-m3`（Ollama Embedding 接口，已开启归一化与查询指令）

## 原则落实
- **KISS**：单文件 Streamlit，无额外前端构建链路。
- **YAGNI**：仅保留问答所需控件，未引入登录/会话持久化等未定义需求。
- **DRY/SOLID**：配置复用 `load_settings`，前后端通过 `RagPipeline` 交互，替换模型无需改 UI 代码。
