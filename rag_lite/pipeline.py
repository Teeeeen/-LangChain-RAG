from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

from .config import RagSettings
from .loaders import load_documents, split_documents
from .vectorstore import build_vector_store, load_vector_store

PROMPT_TEMPLATE = """你是一名简洁的本地 RAG 助手。只使用检索到的上下文用简体中文回答，
如信息不足请回答“不确定”，不要编造。

问题：{question}
上下文：{context}
答案："""


@dataclass
class RagPipeline:
    settings: RagSettings

    def ingest(self) -> Dict[str, int]:
        """Load, split, and index documents; return basic counts."""
        documents = load_documents(self.settings.data_path)
        chunks = split_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        build_vector_store(chunks, self.settings)
        return {"documents": len(documents), "chunks": len(chunks)}

    def ask(self, question: str) -> Dict[str, List[str] | str | Dict[str, float]]:
        docs, retrieval_ms = self._retrieve(question)
        answer, generation_ms = self._generate(question, docs)
        sources = [doc.metadata.get("source", "") for doc in docs]
        return {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
            },
        }

    def _retrieve(self, question: str) -> tuple[List[Document], float]:
        vector_store = load_vector_store(self.settings)
        retriever = vector_store.as_retriever(search_kwargs={"k": self.settings.top_k})
        start = perf_counter()
        documents = retriever.get_relevant_documents(question)
        elapsed = (perf_counter() - start) * 1000
        return documents, elapsed

    def _generate(self, question: str, documents: List[Document]) -> tuple[str, float]:
        llm = self._create_llm()
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["question", "context"]
        )
        context = "\n\n".join(doc.page_content for doc in documents)
        start = perf_counter()
        answer = llm.invoke(prompt.format(question=question, context=context))
        elapsed = (perf_counter() - start) * 1000
        return answer, elapsed

    def _create_llm(self) -> Ollama:
        """Factory for LLM backend (swap here if needed)."""
        return Ollama(
            model=self.settings.llm_model,
            temperature=self.settings.temperature,
            base_url=self.settings.ollama_base_url,
        )
