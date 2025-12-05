# 毕业论文大纲：基于 LangChain 与本地大模型的轻量化 RAG 系统

## 第一章 绪论
1. 研究背景与意义：本地隐私需求、轻量部署趋势。
2. 国内外研究现状：LangChain 生态、Ollama/llamafile 等本地推理方案。
3. 研究内容与结构安排。

## 第二章 相关技术与理论基础
1. LLM + RAG 原理。
2. LangChain 组件（Loader、Text Splitter、Embeddings、VectorStore）。
3. 本地 LLM 部署方式对比：Ollama、LM Studio、vLLM。
4. 轻量化设计原则（KISS、YAGNI、DRY、SOLID）在系统中的体现。

## 第三章 轻量化 RAG 系统架构设计
1. 整体架构：数据层、向量层、推理层、接口层。
2. 模块设计：
   - `config`：统一参数管理（SRP / DIP）。
   - `loaders`：多格式文档加载 & 分块（DRY）。
   - `vectorstore`：Chroma 持久化（OCP）。
   - `pipeline`：检索与生成解耦（KISS）。
   - `cli`：Typer 命令行入口。
3. 流程：数据导入 → 嵌入构建 → 本地问答。

## 第四章 关键技术实现
1. 文档预处理策略与参数选择。
2. 嵌入模型与轻量 LLM 选择依据。
3. 本地推理调用（Ollama）与资源调优。
4. 性能监测方案：`scripts/perf_report.py`。

## 第五章 实验与性能评估
1. 实验环境：硬件、软件、模型版本。
2. 延迟、内存、生成质量指标。
3. 不同模型 / 参数对比实验。
4. 结果分析与优势总结。

## 第六章 总结与展望
1. 研究成果回顾。
2. 面临的挑战：文档规模、回答事实性、GPU 依赖。
3. 后续工作：增量学习、多模态扩展、UI 产品化。

## 附录
1. CLI 使用说明。
2. 关键代码片段。
3. 性能数据原始记录。
