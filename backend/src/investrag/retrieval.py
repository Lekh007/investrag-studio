from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .domain import Chunk
from .embeddings import EmbeddingProvider
from .llm import LocalAnswerer
from .reranker import CrossEncoderReranker
from .vector_adapters import VectorStorePort


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower())


class BM25:
    def __init__(self, records: list[Chunk]) -> None:
        self.records = records
        self.tokenized = [_tokens(record.text) for record in records]
        self.doc_frequency = Counter(token for tokens in self.tokenized for token in set(tokens))
        self.avgdl = sum(map(len, self.tokenized)) / len(self.tokenized) if self.tokenized else 1.0

    def search(self, query: str, k: int = 20, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        query_tokens = _tokens(query)
        n = len(self.records)
        allowed = set(source_ids or [])
        results: list[tuple[Chunk, float]] = []
        for record, tokens in zip(self.records, self.tokenized, strict=True):
            if allowed and record.source_id not in allowed:
                continue
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                if token not in counts:
                    continue
                idf = math.log(1 + (n - self.doc_frequency[token] + 0.5) / (self.doc_frequency[token] + 0.5))
                denominator = counts[token] + 1.5 * (0.25 + 0.75 * len(tokens) / max(self.avgdl, 1))
                score += idf * counts[token] * 2.5 / denominator
            if score > 0:
                results.append((record, score))
        return sorted(results, key=lambda item: item[1], reverse=True)[:k]


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    stage: str

    def document(self) -> Document:
        return Document(page_content=self.chunk.text, metadata={**self.chunk.metadata, "chunk_id": self.chunk.chunk_id, "source_id": self.chunk.source_id})


@dataclass
class RetrievalRun:
    results: list[Retrieved]
    stages: list[dict[str, Any]]


class LangChainRetriever:
    """Explicit Runnable retrieval pipeline; no LangGraph state machine is used."""

    def __init__(
        self,
        store: VectorStorePort,
        embeddings: EmbeddingProvider,
        reranker: CrossEncoderReranker | None = None,
        answerer: LocalAnswerer | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.reranker = reranker
        self.answerer = answerer
        self.chain: Any = RunnablePassthrough() | RunnableLambda(self._run_pipeline)

    @staticmethod
    def _stage(name: str, items: list[Retrieved]) -> dict[str, Any]:
        return {
            "stage": name,
            "candidates": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "source_id": item.chunk.source_id,
                    "rank": rank,
                    "score": round(float(item.score), 6),
                }
                for rank, item in enumerate(items, start=1)
            ],
        }

    def invoke(
        self,
        question: str,
        profile: str,
        k: int,
        source_ids: list[str] | None = None,
    ) -> RetrievalRun:
        return cast(
            RetrievalRun,
            self.chain.invoke(
                {
                    "question": question,
                    "profile": profile,
                    "k": k,
                    "source_ids": source_ids,
                }
            ),
        )

    def search(self, question: str, profile: str, k: int, source_ids: list[str] | None = None) -> list[Retrieved]:
        return self.invoke(question, profile, k, source_ids).results

    def _run_pipeline(self, payload: dict[str, Any]) -> RetrievalRun:
        question = str(payload["question"])
        profile = str(payload["profile"])
        k = int(payload["k"])
        source_ids = payload.get("source_ids")
        if profile == "multi-query":
            template_variants = [question, f"key facts and figures about {question}", f"source evidence for {question}"]
            generation_mode = "extractive-fallback"
            generation_reason: str | None = "no generation model configured for this retriever"
            variants = template_variants
            if self.answerer is not None:
                variant_result = self.answerer.generate_query_variants(question)
                generation_mode = variant_result.mode
                generation_reason = variant_result.fallback_reason
                if variant_result.mode == "model":
                    variants = [question, *variant_result.variants]
            merged: dict[str, Retrieved] = {}
            for variant in variants:
                variant_run = self._run_pipeline(
                    {
                        "question": variant,
                        "profile": "hybrid",
                        "k": k * 2,
                        "source_ids": source_ids,
                    }
                )
                for rank, item in enumerate(variant_run.results, start=1):
                    if item.chunk.chunk_id not in merged:
                        merged[item.chunk.chunk_id] = Retrieved(item.chunk, 0.0, "multi-query")
                    merged[item.chunk.chunk_id].score += 1 / (60 + rank)
            results = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:k]
            variant_stage = {
                "stage": "multi-query-generation",
                "generation_mode": generation_mode,
                "generation_fallback_reason": generation_reason,
                "variant_count": len(variants),
            }
            return RetrievalRun(results, [variant_stage, self._stage("multi-query-fusion", results)])
        query_vector = self.embeddings.embed([question]).vectors[0]
        dense = [Retrieved(chunk, score, "dense") for chunk, score in self.store.search(query_vector, k=max(20, k * 4), source_ids=source_ids)]
        stages = [self._stage("dense", dense)]
        if profile == "dense":
            results = dense[:k]
            return RetrievalRun(results, stages + [self._stage("final", results)])
        if profile == "parent":
            # design §9.3/§11 step 7: search small chunks, return full parent
            # sections as context. Real ``parent-child``-profile chunks resolve
            # `parent_id` to an actual stored parent chunk, which is fetched and
            # returned in place of the child; ``structure-aware`` chunks instead
            # point `parent_id` at a heading *element* id (never a chunk id in the
            # store), so that case falls back to the previous dedup-only behaviour
            # rather than expanding to nothing.
            chunk_by_id = {chunk.chunk_id: chunk for chunk in self.store.all_records()}
            parent_selected: list[Retrieved] = []
            seen_parents: set[str] = set()
            for candidate in dense:
                parent_key = candidate.chunk.parent_id or candidate.chunk.chunk_id
                if parent_key in seen_parents:
                    continue
                seen_parents.add(parent_key)
                parent_chunk = chunk_by_id.get(candidate.chunk.parent_id) if candidate.chunk.parent_id else None
                expanded = Retrieved(parent_chunk, candidate.score, "parent") if parent_chunk is not None else candidate
                parent_selected.append(expanded)
                if len(parent_selected) >= k:
                    break
            return RetrievalRun(parent_selected, stages + [self._stage("parent-expansion", parent_selected)])
        if profile == "mmr":
            pool = dense[: max(20, k * 4)]
            mmr_selected: list[Retrieved] = []
            stored_vector_for = getattr(self.store, "vector_for", lambda _chunk_id: None)
            fallback_vectors: dict[str, Any] = {}
            if pool:
                embedded_pool = self.embeddings.embed([item.chunk.text for item in pool]).vectors
                fallback_vectors = {
                    item.chunk.chunk_id: vector
                    for item, vector in zip(pool, embedded_pool, strict=True)
                }

            def vector_for(chunk_id: str) -> Any:
                stored = stored_vector_for(chunk_id)
                return stored if stored is not None else fallback_vectors.get(chunk_id)

            while pool and len(mmr_selected) < k:
                def mmr_score(candidate: Retrieved) -> float:
                    candidate_vector = vector_for(candidate.chunk.chunk_id)
                    similarities = [
                        float(candidate_vector @ selected_vector)
                        for item in mmr_selected
                        if candidate_vector is not None
                        and (selected_vector := vector_for(item.chunk.chunk_id)) is not None
                    ]
                    diversity = max(similarities, default=0.0)
                    return 0.75 * candidate.score - 0.25 * diversity if mmr_selected else candidate.score

                best = max(pool, key=mmr_score)
                mmr_selected.append(best)
                pool.remove(best)
            return RetrievalRun(mmr_selected, stages + [self._stage("mmr", mmr_selected)])
        lexical = [Retrieved(chunk, score, "bm25") for chunk, score in BM25(self.store.all_records()).search(question, k=max(20, k * 4), source_ids=source_ids)]
        stages.append(self._stage("bm25", lexical))
        by_id: dict[str, Retrieved] = {}
        for rank, item in enumerate(dense, start=1):
            by_id.setdefault(item.chunk.chunk_id, Retrieved(item.chunk, 0.0, "rrf"))
            by_id[item.chunk.chunk_id].score += 1 / (60 + rank)
        for rank, item in enumerate(lexical, start=1):
            by_id.setdefault(item.chunk.chunk_id, Retrieved(item.chunk, 0.0, "rrf"))
            by_id[item.chunk.chunk_id].score += 1 / (60 + rank)
        # Production profile (design §11): fuse to a wide pool (top 20), rerank it,
        # THEN truncate to the requested top_k - not the other way around, or
        # reranking would have nothing but the already-narrow top_k to work with.
        fused_pool = sorted(by_id.values(), key=lambda item: item.score, reverse=True)[: max(20, k)]
        stages.append(self._stage("rrf", fused_pool))
        if self.reranker is not None and fused_pool:
            rerun = self.reranker.rerank(question, [(item.chunk.chunk_id, item.chunk.text) for item in fused_pool])
            by_chunk_id = {item.chunk.chunk_id: item for item in fused_pool}
            reranked = [Retrieved(by_chunk_id[r.chunk_id].chunk, r.score, "rerank") for r in rerun.results if r.chunk_id in by_chunk_id]
            stages.append({**self._stage("rerank", reranked), "used_reranker": rerun.used_reranker, "reason": rerun.reason})
            results = reranked[:k] if rerun.used_reranker else fused_pool[:k]
        else:
            results = fused_pool[:k]
        return RetrievalRun(results, stages + [self._stage("final", results)])
