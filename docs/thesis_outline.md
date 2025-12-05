# 论文大纲

## 选题依据及意义（不少于300字）
随着大模型本地化和开源化的发展，传统依赖云端算力的问答系统在数据安全、延迟和成本上逐渐暴露局限。本课题聚焦“基于 LangChain 与本地大模型的轻量化 RAG 系统”，选题意义体现在：第一，数据主权与隐私合规需求高涨，医疗、政务、企业内训资料等敏感内容更适合在本地完成检索与生成，避免云端泄露风险；第二，轻量化与量化技术（如 8B 级别的 Qwen3、bge-m3 轻量嵌入）使普通 PC/工作站具备可行的推理性能，降低部署门槛；第三，RAG（Retrieval-Augmented Generation）通过检索补充上下文，显著降低幻觉率，相比纯生成模型更契合知识型应用；第四，LangChain 等统一编排框架降低了组件拼装和替换成本，便于快速迭代。本课题旨在验证“在普通 PC 环境下，用轻量模型与本地向量库即可实现低成本、高可控的问答”，为后续行业落地提供工程实践参考。

## 研究现状及发展态势（不少于300字）
当前 RAG 研究从“模型能力”与“检索增强”双向演进：一方面，国内外开源模型（如 LLaMA 系列、Qwen、Baichuan）在指令遵循与对话质量上快速逼近商用闭源模型，并通过量化与蒸馏扩展到消费级硬件。另一方面，检索侧引入更强的嵌入模型（如 bge-m3、多语言文本向量）、稀疏与稠密混合检索（BM25+向量）、重排序（cross-encoder）以提升召回质量。工具链层面，LangChain、LlamaIndex 等框架提供了加载、切分、向量化、检索、生成的流水线抽象；向量库如 Chroma、Milvus、Weaviate 提供本地或云端的弹性存储。行业趋势从“单一模型问答”转向“可插拔组件 + 可观测 + 评测闭环”，并逐渐重视成本、隐私与可维护性。本课题选择普通 PC 环境 + 本地 Ollama 推理，验证轻量栈在真实硬件上的可行性，同时结合 RAGAS 等评测手段量化生成质量，补齐工程侧的可复现基线。

## 课题研究内容、拟解决的关键问题和最终目标（不少于500字）
研究内容围绕“轻量化、本地化、可评测”的 RAG 系统构建与验证：
1. 模型与嵌入选择：评估适配普通 PC 的轻量 LLM（Qwen3:8b）与嵌入模型（bge-m3），兼顾生成质量与资源占用；通过 Ollama 统一管理模型，降低部署复杂度。
2. 数据处理与向量化：实现文档加载（文本/PDF）、分块（递归分词、分句）、向量化（归一化、查询指令）与向量库构建（Chroma 持久化），确保处理流程可配置且复用。
3. 检索与生成：构建统一的 RagPipeline，串联检索（TopK）、提示模板与 LLM 生成；暴露 CLI、FastAPI、Streamlit 前端多种入口，以便开发与演示。
4. 评测与观测：在普通 PC 环境运行性能脚本，记录检索/生成耗时、内存占用；引入 RAGAS（本地 LLM/Embedding 适配）评估忠实度、相关性、正确性，形成质量报告模板。
拟解决的关键问题包括：如何在有限算力下保持响应速度与生成质量的平衡；如何通过检索增强降低幻觉；如何以最少依赖、最小改动实现组件可插拔（如更换模型、向量库）；如何在无外部 API 的条件下完成可重复的质量评估。最终目标是交付一个可在普通 PC/WSL2 上运行的本地 RAG 原型，具备：1）可配置的 ingest/ask 流程；2）可观测的性能与质量指标；3）简洁的前端交互；4）完善的文档与报告模板，便于复现与扩展。

## 拟采取的主要技术路线、实施方案和工具等（不少于500字）
技术路线遵循“简单、可插拔、可验证”三原则：
1. 架构与框架：采用 LangChain 作为编排框架，封装 RagPipeline（ingest/ask），确保加载、切分、检索、生成的单一职责；使用 FastAPI 提供 REST 接口，Streamlit 提供轻量前端。
2. 模型与推理：使用 Ollama 管理并运行 `qwen3:8b`、`bge-m3`，支持 CPU/GPU（WSL2 可用）；通过配置注入模型名与 base_url，便于切换或远程部署。
3. 数据与向量库：数据加载支持 txt/md/pdf；分块采用 RecursiveCharacterTextSplitter；向量化支持 Ollama Embedding 与本地 HuggingFace，默认归一化与查询指令；向量库使用 Chroma 持久化至本地目录。
4. 评测与观测：性能脚本收集检索/生成耗时、总耗时、峰值 RSS；质量评测引入 RAGAS，适配本地 LLM/Embedding，无需外部 API；提供报告模板与命令示例，默认运行 10 次以平滑波动。
5. 工具与环境：Python 3.10+；依赖管理通过 requirements.txt；开发/调试使用 WSL2 + conda + Ollama；前端通过 Streamlit 即时运行；命令行工具 Typer；日志与错误提示保持简洁。
实施方案：
- 阶段1：环境与依赖安装，拉取模型，构建向量库；验证 ingest/ask 基本链路。
- 阶段2：前端与 API 联调，确保可配置性与可视化输出（来源、耗时）。
- 阶段3：性能与质量评测，产出报告与改进建议（如调节 TopK、chunk_size）。
- 阶段4：文档与论文大纲完善，提供复现步骤、配置项、评测模板。
工具：LangChain、Chroma、Ollama、Streamlit、FastAPI、Typer、RAGAS、jieba、datasets/evaluate（RAGAS 依赖）、psutil（资源监测）。

## 主要参考文献（不少于10篇）
1. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP. NeurIPS.
2. Izacard, G. & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain QA. EACL.
3. Gao, L. et al. (2023). RAGAS: Automated Evaluation of Retrieval-Augmented Generation. arXiv:2309.15218.
4. Chen, M. et al. (2024). BGE-M3: Multi-Functional Multi-Lingual Embeddings. arXiv.
5. Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv.
6. Bai, J. et al. (2023). Qwen Technical Report. Alibaba Group. arXiv.
7. Zhang, X. et al. (2022). A Survey on Retrieval-Augmented Text Generation. ACL Anthology.
8. Sun, Y. et al. (2023). Evaluating and Mitigating Hallucinations in LLMs. ACL/EMNLP.
9. Guo, J. et al. (2022). A Deep Look into Neural Ranking Models. IJCAI.
10. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
11. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
12. Shen, Y. et al. (2023). MedQA: A Large-Scale Medical QA Benchmark for Chinese. ACL Anthology.
