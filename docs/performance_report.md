# 轻量化 RAG 性能报告（模板）

说明：面向 `qwen3:8b`（Ollama）+ `ollama:bge-m3`（Ollama Embedding）组合的本地评测模板。请运行脚本获取实测数据后填入本文件。

## 运行命令
```powershell
python scripts/perf_report.py `
  --question "RAG 管线的瓶颈是什么？" `
  --runs 3 `
  --persist-dir data/vector_store `
  --embedding ollama:bge-m3 `
  --model qwen3:8b
```

## 测试环境（请填写）
- 硬件：如 Intel/AMD CPU，内存，显卡（显存）
- 操作系统：如 Windows 11 / Ubuntu 22.04
- Python 版本：
- 语料：示例 `data/sample_docs`
- Chunk 设置：`chunk_size=500`，`chunk_overlap=80`
- Ollama：`RAG_OLLAMA_BASE_URL`（如有自定义）

## 结果示例（请替换为实测值）
| run | retrieval_ms | generation_ms | total_ms | peak_RSS_MB |
| --- | ------------ | ------------- | -------- | ----------- |
| 1   | ...          | ...           | ...      | ...         |
| 2   | ...          | ...           | ...      | ...         |
| 3   | ...          | ...           | ...      | ...         |
| 平均 | ...         | ...           | ...      | ...         |

## 观察（示例）
- KISS：单进程即可得到 <1s 级响应，无需额外服务。
- YAGNI：在小语料下未启用 rerank/缓存，保持最少组件便于调优。
- DRY/SOLID：性能脚本复用 `RagPipeline`，避免重复实现计时与调用逻辑。
