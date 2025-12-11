from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
import sys

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_lite.config import load_settings
from rag_lite.pipeline import RagPipeline


def human_mb(value_bytes: float) -> float:
    return round(value_bytes / (1024 * 1024), 2)


def run_once(pipeline: RagPipeline, question: str) -> dict:
    result = pipeline.ask(question)
    metrics = result["metrics"]
    retrieval_ms = metrics["retrieval_ms"]
    generation_ms = metrics["generation_ms"]
    total = retrieval_ms + generation_ms
    return {
        "answer": result["answer"],
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total,
        "sources": result["sources"],
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 性能评估")
    parser.add_argument("--question", required=True, help="测试问题")
    parser.add_argument(
        "--runs", type=int, default=10, help="重复次数（默认 10，可按需调整）"
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/vector_store"),
        help="向量库目录",
    )
    parser.add_argument(
        "--embedding",
        default="ollama:bge-m3",
        help="Embedding 模型名称",
    )
    parser.add_argument("--model", default="qwen3:8b", help="LLM 模型名称")
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama 服务地址（默认 http://localhost:11434）",
    )
    args = parser.parse_args()

    settings = load_settings(
        persist_dir=args.persist_dir,
        embedding_model=args.embedding,
        llm_model=args.model,
        ollama_base_url=args.ollama_base_url,
    )
    pipeline = RagPipeline(settings)
    process = psutil.Process()

    total_metrics = []
    mem_snapshots = []
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    for idx in range(1, args.runs + 1):
        before = process.memory_info().rss
        run_metric = run_once(pipeline, args.question)
        after = process.memory_info().rss
        mem_snapshots.append(max(before, after))
        total_metrics.append(run_metric)
        print(
            f"[第 {idx} 轮] 检索={run_metric['retrieval_ms']:.1f} ms | "
            f"生成={run_metric['generation_ms']:.1f} ms | "
            f"总计={run_metric['total_ms']:.1f} ms"
        )

    avg_total = statistics.fmean(m["total_ms"] for m in total_metrics)
    avg_retrieval = statistics.fmean(m["retrieval_ms"] for m in total_metrics)
    avg_generation = statistics.fmean(m["generation_ms"] for m in total_metrics)
    peak_mem = human_mb(max(mem_snapshots))

    print("\n=== 性能汇总 ===")
    print(f"时间戳        : {start_ts}")
    print(f"问题          : {args.question}")
    print(f"轮次          : {args.runs}")
    print(f"Embedding 模型: {args.embedding}")
    print(f"LLM 模型      : {args.model}")
    print(f"平均检索耗时  : {avg_retrieval:.1f} ms")
    print(f"平均生成耗时  : {avg_generation:.1f} ms")
    print(f"平均总耗时    : {avg_total:.1f} ms")
    print(f"峰值常驻内存  : {peak_mem} MB")
    print(f"向量库目录    : {args.persist_dir}")

    print("\n示例回答:")
    print(total_metrics[-1]["answer"])
    print("\n来源:")
    for source in total_metrics[-1]["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
