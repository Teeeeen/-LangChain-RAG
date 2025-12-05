from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
import sys
import re

import jieba
import psutil
from langchain_community.chat_models import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_lite.config import load_settings
from rag_lite.pipeline import RagPipeline
from rag_lite.vectorstore import create_embeddings


def human_mb(value_bytes: float) -> float:
    return round(value_bytes / (1024 * 1024), 2)


def run_once(pipeline: RagPipeline, question: str) -> dict:
    docs, retrieval_ms = pipeline._retrieve(question)
    answer, generation_ms = pipeline._generate(question, docs)
    sources = [doc.metadata.get("source", "") for doc in docs]
    total = retrieval_ms + generation_ms
    return {
        "answer": answer,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total,
        "sources": sources,
        "contexts": [doc.page_content for doc in docs],
    }


def _tokens(text: str) -> set[str]:
    """
    Tokenize mixed Chinese/English text.
    - Chinese: jieba 分词
    - English/数字：简单按空格/标点切分
    """
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    zh_tokens = [t.strip() for t in jieba.lcut(cleaned) if t.strip()]
    # 保留英文/数字的粗粒度切分，避免全被当作单个 token
    en_tokens = [t for t in cleaned.split() if t]
    return set(zh_tokens + en_tokens)


def evaluate_quality(
    pipeline: RagPipeline, question: str, answer: str, ground_truth: str | None
) -> dict:
    """
    RAGAS-inspired heuristic metrics:
    - faithfulness: answer tokens vs retrieved context tokens
    - relevance: answer tokens vs question tokens
    - correctness: answer tokens vs provided ground-truth tokens (if any)
    """
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)

    docs, _ = pipeline._retrieve(question)
    context_tokens: set[str] = set()
    for doc in docs:
        context_tokens |= _tokens(doc.page_content)

    faithfulness = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )
    relevance = (
        len(answer_tokens & question_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    correctness = None
    if ground_truth:
        gt_tokens = _tokens(ground_truth)
        correctness = (
            len(answer_tokens & gt_tokens) / len(answer_tokens)
            if answer_tokens
            else 0.0
        )

    return {
        "faithfulness": round(faithfulness, 3),
        "relevance": round(relevance, 3),
        "correctness": round(correctness, 3) if correctness is not None else "N/A",
    }


def compute_ragas_metrics(
    settings,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None,
) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, answer_correctness, faithfulness
        from ragas.embeddings import LangchainEmbeddings
    except Exception:
        return None

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    metrics = [faithfulness, answer_relevancy]
    if ground_truth:
        data["ground_truth"] = [ground_truth]
        metrics.append(answer_correctness)

    ds = Dataset.from_dict(data)
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )
    embeddings = LangchainEmbeddings(
        create_embeddings(settings.embedding_model, settings.ollama_base_url)
    )
    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)

    if hasattr(result, "scores"):
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in result.scores.items()}
    if isinstance(result, dict):
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}
    return None


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
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="可选：用于正确性评估的标准答案文本",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=None,
        help="可选：包含标准答案的文件路径",
    )
    args = parser.parse_args()

    ground_truth = args.ground_truth
    if args.ground_truth_file:
        ground_truth = args.ground_truth_file.read_text(encoding="utf-8")

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

    quality = evaluate_quality(
        pipeline, args.question, total_metrics[-1]["answer"], ground_truth
    )
    ragas_scores = compute_ragas_metrics(
        settings,
        args.question,
        total_metrics[-1]["answer"],
        total_metrics[-1]["contexts"],
        ground_truth,
    )

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
    print("Quality (RAGAS-style heuristic):")
    print(f"- Faithfulness: {quality['faithfulness']}")
    print(f"- Answer Relevance: {quality['relevance']}")
    print(f"- Correctness: {quality['correctness']}")
    if ragas_scores:
        print("Quality (RAGAS model metrics):")
        for k, v in ragas_scores.items():
            print(f"- {k}: {v}")
    else:
        print("Quality (RAGAS model metrics): skipped (ragas not installed)")

    print("\nSample answer:")
    print(total_metrics[-1]["answer"])
    print("\nSources:")
    for source in total_metrics[-1]["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
