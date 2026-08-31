"""Run the deterministic InvestRAG portfolio demonstration without Ollama."""

from __future__ import annotations

import argparse
from pathlib import Path

from create_demo_corpus import create_corpus

from investrag.config import Settings
from investrag.domain import ExperimentManifest, GoldenDataset, GoldenQuestion
from investrag.evaluation import ExperimentRunner
from investrag.service import InvestRAGService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(".data/demo-corpus"))
    parser.add_argument("--data-dir", type=Path, default=Path(".data/investrag-demo"))
    args = parser.parse_args()
    paths = create_corpus(args.corpus)
    service = InvestRAGService(Settings(data_dir=args.data_dir, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    sources = [service.ingest(path) for path in paths]
    brief = next(source for source in sources if source.name == "brief.md")
    relevant = next(chunk.chunk_id for chunk in service.catalog.list_chunks() if chunk.source_id == brief.source_id and "Revenue grew" in chunk.text)
    dataset = GoldenDataset(dataset_id="synthetic-smoke", name="Synthetic smoke fixture (not a reviewed benchmark)", questions=[GoldenQuestion(question_id="q1", question="What happened to revenue?", relevant_chunk_ids=[relevant], notes="Programmatically linked for pipeline smoke testing only")], corpus_source_ids=[brief.source_id])
    service.catalog.save_golden_dataset(dataset)
    record = ExperimentRunner(service).run(dataset, ExperimentManifest(dataset_id=dataset.dataset_id, vector_stores=["faiss", "chroma", "qdrant"], profiles=["hybrid"], top_k=3))
    service.catalog.save_experiment(record)
    print(f"ingested_sources={len(sources)} indexed_chunks={service.store.count}")
    print(f"experiment={record.experiment_id} status={record.status} results={len(record.results)}")
    for metric in record.metrics:
        if metric.name.endswith("/success@3") or metric.name.endswith("/ndcg@3"):
            print(f"{metric.name}={metric.value}")
    if record.warnings:
        print("warnings:")
        for warning in record.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
