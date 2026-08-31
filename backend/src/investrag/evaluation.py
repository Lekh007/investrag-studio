from __future__ import annotations

import hashlib
import math
import platform
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from statistics import median
from typing import TYPE_CHECKING, Literal

from .domain import (
    ExperimentManifest,
    ExperimentQuestionResult,
    ExperimentRecord,
    GoldenDataset,
    MetricResult,
)


def success_at_k(relevant: set[str], retrieved: Iterable[str], k: int) -> float:
    return float(bool(relevant.intersection(list(retrieved)[:k])))


def recall_at_k(relevant: set[str], retrieved: Iterable[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(list(retrieved)[:k])) / len(relevant)


def reciprocal_rank(relevant: set[str], retrieved: Iterable[str], k: int) -> float:
    for index, item in enumerate(list(retrieved)[:k], start=1):
        if item in relevant:
            return 1 / index
    return 0.0


def ndcg_at_k(relevance: dict[str, int], retrieved: Iterable[str], k: int) -> float:
    ranked = list(retrieved)[:k]
    dcg = sum((2 ** relevance.get(item, 0) - 1) / math.log2(index + 2) for index, item in enumerate(ranked))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def citation_coverage(answer: str, cited_labels: set[str]) -> float:
    citations = {token.strip("[]") for token in answer.split() if token.startswith("[") and token.endswith("]")}
    return float(bool(citations and citations.intersection(cited_labels))) if cited_labels else 0.0


class ExperimentRunner:
    """Run deterministic retrieval experiments over reviewed golden judgments.

    Generation is deliberately excluded from the default benchmark so results remain
    reproducible without an Ollama process. A future judge track can consume the same
    manifest and add clearly labelled local-LLM metrics.
    """

    def __init__(self, service: InvestRAGService) -> None:
        self.service = service

    def run(self, dataset: GoldenDataset, manifest: ExperimentManifest) -> ExperimentRecord:
        if manifest.track != "portable":
            raise ValueError("only the portable experiment track is implemented")
        started_at = datetime.now(UTC)
        warnings: list[str] = []
        question_results: list[ExperimentQuestionResult] = []
        canonical_chunks = self.service.store.all_records()
        by_element: dict[str, set[str]] = {}
        for chunk in canonical_chunks:
            for element_id in chunk.element_ids:
                by_element.setdefault(element_id, set()).add(chunk.chunk_id)
        if not dataset.questions:
            warnings.append("golden dataset contains no questions")

        attempted_stores = 0
        for store_name in manifest.vector_stores:
            try:
                store = self.service.registry.get(store_name)
            except RuntimeError as exc:
                warnings.append(f"{store_name}: unavailable ({exc})")
                continue
            attempted_stores += 1
            if store.count == 0:
                warnings.append(f"{store_name}: collection is empty; ingest and publish the corpus first")
            from .retrieval import LangChainRetriever

            retriever = LangChainRetriever(store, self.service.embeddings)
            for profile_name in manifest.profiles:
                for question in dataset.questions:
                    relevant = set(question.relevant_chunk_ids)
                    if not relevant and question.relevant_element_ids:
                        for element_id in question.relevant_element_ids:
                            relevant.update(by_element.get(element_id, set()))
                    if question.answerable and not relevant:
                        warnings.append(f"{question.question_id}: no relevant chunk/element judgment")
                    for _ in range(manifest.warmup_runs):
                        retriever.search(question.question, profile_name, manifest.top_k, dataset.corpus_source_ids or None)
                    repetitions: list[tuple[list[str], float]] = []
                    for _ in range(manifest.repetitions):
                        query_started = time.perf_counter()
                        retrieval_run = retriever.invoke(question.question, profile_name, manifest.top_k, dataset.corpus_source_ids or None)
                        retrieved = self.service.filter_supported_results(
                            question.question, retrieval_run.results
                        )
                        latency_ms = (time.perf_counter() - query_started) * 1000
                        repetitions.append(([item.chunk.chunk_id for item in retrieved], latency_ms))
                    retrieved_ids = repetitions[0][0]
                    if any(ids != retrieved_ids for ids, _ in repetitions[1:]):
                        warnings.append(f"{question.question_id}: retrieval ranks changed across repetitions")
                    latency_samples = [latency for _, latency in repetitions]
                    latency_ms = median(latency_samples)
                    predicted_abstention = not bool(retrieved_ids)
                    question_results.append(
                        ExperimentQuestionResult(
                            question_id=question.question_id,
                            vector_store=store_name,
                            profile=profile_name,
                            retrieved_chunk_ids=retrieved_ids,
                            relevant_chunk_ids=sorted(relevant),
                            success_at_k=success_at_k(relevant, retrieved_ids, manifest.top_k),
                            recall_at_k=recall_at_k(relevant, retrieved_ids, manifest.top_k),
                            reciprocal_rank=reciprocal_rank(relevant, retrieved_ids, manifest.top_k),
                            ndcg_at_k=ndcg_at_k({chunk_id: 1 for chunk_id in relevant}, retrieved_ids, manifest.top_k),
                            retrieval_ms=round(latency_ms, 3),
                            retrieval_samples_ms=[round(value, 3) for value in latency_samples],
                            predicted_abstention=predicted_abstention,
                            abstention_correct=(predicted_abstention == (not question.answerable)),
                        )
                    )

        metrics: list[MetricResult] = []
        groups: dict[tuple[str, str], list[ExperimentQuestionResult]] = {}
        for result in question_results:
            groups.setdefault((result.vector_store, result.profile), []).append(result)
        for (store_name, profile), results in groups.items():
            prefix = f"{store_name}/{profile}"
            # An unanswerable question has no relevant_chunk_ids by construction, so
            # success/recall/ndcg are mathematically 0 for it regardless of retrieval
            # quality - blending them into these metrics silently punishes a system
            # for correctly abstaining. Found via a real golden set that includes the
            # design §14.1-mandated unanswerable class, 2026-08-30: a perfectly
            # correct 13/13 answerable-question run still read as "86.7% success"
            # before this filter. Design §14.7 keeps "answerable-question accuracy"
            # and "abstention accuracy" as separate gates for exactly this reason.
            answerable_results = [item for item in results if item.relevant_chunk_ids]
            metrics.extend(
                [
                    MetricResult(name=f"{prefix}/success@{manifest.top_k}", value=_mean(item.success_at_k for item in answerable_results)),
                    MetricResult(name=f"{prefix}/recall@{manifest.top_k}", value=_mean(item.recall_at_k for item in answerable_results)),
                    MetricResult(name=f"{prefix}/mrr@{manifest.top_k}", value=_mean(item.reciprocal_rank for item in answerable_results)),
                    MetricResult(name=f"{prefix}/ndcg@{manifest.top_k}", value=_mean(item.ndcg_at_k for item in answerable_results)),
                    MetricResult(name=f"{prefix}/abstention-accuracy", value=_mean(float(item.abstention_correct) for item in results if item.abstention_correct is not None)),
                    MetricResult(name=f"{prefix}/retrieval-latency-p95-ms", value=_percentile([item.retrieval_ms for item in results], 0.95), unit="milliseconds"),
                ]
            )
        status: Literal["completed", "partial", "failed"] = "completed" if attempted_stores and not warnings else "partial" if attempted_stores else "failed"
        completed_at = datetime.now(UTC)
        return ExperimentRecord(
            experiment_id=f"exp_{uuid.uuid4().hex[:12]}",
            manifest=manifest,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            corpus_chunk_count=len(canonical_chunks),
            results=question_results,
            metrics=metrics,
            warnings=sorted(set(warnings)),
            reproducibility=self._reproducibility(dataset, manifest),
        )

    def _reproducibility(
        self, dataset: GoldenDataset, manifest: ExperimentManifest
    ) -> dict[str, object]:
        selected_ids = set(dataset.corpus_source_ids)
        sources = [
            source
            for source in self.service.list_sources()
            if not selected_ids or source.source_id in selected_ids
        ]
        corpus_material = "\n".join(
            f"{source.source_id}:{source.checksum}" for source in sorted(sources, key=lambda item: item.source_id)
        )
        dependencies: dict[str, str] = {}
        for package in ("numpy", "faiss-cpu", "chromadb", "qdrant-client", "langchain-core"):
            try:
                dependencies[package] = version(package)
            except PackageNotFoundError:
                dependencies[package] = "not-installed"
        return {
            "schema_version": "1",
            "corpus_checksum": hashlib.sha256(corpus_material.encode()).hexdigest(),
            "source_version_ids": [source.source_id for source in sources],
            "source_checksums": {source.source_id: source.checksum for source in sources},
            "embedding_model": self.service.embeddings.active_model_name,
            "embedding_dimension": self.service.embeddings.dimension,
            "embedding_fallback": self.service.embeddings.fallback,
            "chunk_profiles": list(manifest.profiles),
            "retrieval": {
                "top_k": manifest.top_k,
                "warmup_runs": manifest.warmup_runs,
                "repetitions": manifest.repetitions,
                "rrf_k": 60,
                "evidence_gate": "non-stopword lexical overlap",
            },
            "vector_stores": list(manifest.vector_stores),
            "seed": "not-applicable: deterministic exact/local retrieval",
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependencies,
        }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 3)


@dataclass(frozen=True)
class IrMeasuresCrossCheck:
    """Design §14.3: `ir-measures` is the citable, independent reference
    implementation for Success/Recall/RR/nDCG. This project's own hand-rolled
    versions above are what the serving and evaluation paths actually run (no
    dependency on an external library on the query hot path), but every completed
    experiment can be cross-checked against `ir-measures` to prove the two agree -
    a real verification, not an assumption that a from-scratch implementation of a
    well-known metric is correct."""

    available: bool
    max_absolute_difference: float
    agrees: bool
    reason: str | None = None


def cross_check_with_ir_measures(record: ExperimentRecord, tolerance: float = 1e-6) -> IrMeasuresCrossCheck:
    try:
        import ir_measures
        from ir_measures import RR, Recall, Success, nDCG
    except ImportError as exc:
        return IrMeasuresCrossCheck(available=False, max_absolute_difference=0.0, agrees=False, reason=f"ir_measures not installed: {exc}")

    k = record.manifest.top_k
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for result in record.results:
        query_id = f"{result.vector_store}/{result.profile}/{result.question_id}"
        qrels[query_id] = dict.fromkeys(result.relevant_chunk_ids, 1)
        run[query_id] = {chunk_id: float(len(result.retrieved_chunk_ids) - rank) for rank, chunk_id in enumerate(result.retrieved_chunk_ids)}

    if not qrels:
        return IrMeasuresCrossCheck(available=True, max_absolute_difference=0.0, agrees=True, reason="no results to compare")

    try:
        ir_aggregate = ir_measures.calc_aggregate([Success @ k, Recall @ k, RR @ k, nDCG @ k], qrels, run)
    except Exception as exc:  # a cross-check failure must not fail the experiment itself
        return IrMeasuresCrossCheck(available=True, max_absolute_difference=0.0, agrees=False, reason=f"ir_measures computation failed: {exc}")

    ir_means = {
        "success": float(ir_aggregate[Success @ k]),
        "recall": float(ir_aggregate[Recall @ k]),
        "rr": float(ir_aggregate[RR @ k]),
        "ndcg": float(ir_aggregate[nDCG @ k]),
    }
    # ir_measures.calc_aggregate averages over every query_id in one flat pool
    # regardless of (store, profile) grouping, so the comparison must average this
    # project's own per-question metrics the same way - a single flat mean across
    # every result, not a per-group mean. Uses an *unrounded* mean, unlike _mean()
    # (which rounds to 4dp for display) - rounding first would compare against a
    # display artifact, not the actual computed values.
    count = len(record.results)
    own_flat = {
        "success": sum(r.success_at_k for r in record.results) / count,
        "recall": sum(r.recall_at_k for r in record.results) / count,
        "rr": sum(r.reciprocal_rank for r in record.results) / count,
        "ndcg": sum(r.ndcg_at_k for r in record.results) / count,
    }
    differences = {metric: abs(own_flat[metric] - ir_means[metric]) for metric in own_flat}
    max_difference = max(differences.values())
    return IrMeasuresCrossCheck(available=True, max_absolute_difference=round(max_difference, 6), agrees=max_difference <= tolerance)


if TYPE_CHECKING:
    from .service import InvestRAGService
