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
    total = metrics["retrieval_ms"] + metrics["generation_ms"]
    return {
        "answer": result["answer"],
        "retrieval_ms": metrics["retrieval_ms"],
        "generation_ms": metrics["generation_ms"],
        "total_ms": total,
        "sources": result["sources"],
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 性能评估")
    parser.add_argument("--question", required=True, help="测试问题")
    parser.add_argument("--runs", type=int, default=3, help="重复次数")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/vector_store"),
        help="向量库目录",
    )
    parser.add_argument(
        "--embedding",
        default="BAAI/bge-m3",
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
            f"[run {idx}] retrieval={run_metric['retrieval_ms']:.1f} ms | "
            f"generation={run_metric['generation_ms']:.1f} ms | "
            f"total={run_metric['total_ms']:.1f} ms"
        )

    avg_total = statistics.fmean(m["total_ms"] for m in total_metrics)
    avg_retrieval = statistics.fmean(m["retrieval_ms"] for m in total_metrics)
    avg_generation = statistics.fmean(m["generation_ms"] for m in total_metrics)
    peak_mem = human_mb(max(mem_snapshots))

    print("\n=== Performance Summary ===")
    print(f"Timestamp      : {start_ts}")
    print(f"Question       : {args.question}")
    print(f"Runs           : {args.runs}")
    print(f"Embedding Model: {args.embedding}")
    print(f"LLM Model      : {args.model}")
    print(f"Avg Retrieval  : {avg_retrieval:.1f} ms")
    print(f"Avg Generation : {avg_generation:.1f} ms")
    print(f"Avg Total      : {avg_total:.1f} ms")
    print(f"Peak RSS       : {peak_mem} MB")
    print(f"Persist Dir    : {args.persist_dir}")

    print("\nSample answer:")
    print(total_metrics[-1]["answer"])
    print("\nSources:")
    for source in total_metrics[-1]["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
