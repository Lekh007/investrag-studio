"""CatalogPort tests: SqliteCatalog (offline default) and PostgresCatalog (design
§4's real metadata store, gated on the docker-compose pgvector service). Both
backends run the SAME test bodies against the SAME port to prove they are
interchangeable, not just individually functional."""

from __future__ import annotations

import pytest

from investrag.catalog import CatalogPort, PostgresCatalog, SqliteCatalog, create_catalog
from investrag.domain import (
    Chunk,
    ExperimentManifest,
    ExperimentRecord,
    GoldenDataset,
    QueryTrace,
    SourceVersion,
    utc_now,
)
from investrag.vector_adapters import AdapterUnavailable

_PG_DSN = "postgresql://investrag:investrag@127.0.0.1:5433/investrag"


def _postgres_or_skip() -> PostgresCatalog:
    try:
        return PostgresCatalog(_PG_DSN)
    except AdapterUnavailable as exc:
        pytest.skip(f"pgvector service unreachable: {exc}")


def _source(source_id: str = "src_1") -> SourceVersion:
    return SourceVersion(logical_source_id="logical_1", version_id=source_id, source_id=source_id, name="brief.md", media_type="text/markdown", checksum="abc123", size_bytes=42, parser="native-text", status="ready", quality_score=1.0, element_count=1, chunk_count=1, queryable=True)


def _run_catalog_contract(catalog: CatalogPort) -> None:
    source = _source()
    assert catalog.get_source(source.source_id) is None
    catalog.save_source(source)
    assert catalog.get_source(source.source_id) == source
    assert catalog.list_sources() == [source]

    chunk = Chunk(chunk_id="chunk_1", source_id=source.source_id, profile="structure-aware", text="Revenue grew.", element_ids=["e1"])
    catalog.save_chunks([chunk])
    assert catalog.list_chunks() == [chunk]

    dataset = GoldenDataset(dataset_id="ds_1", name="Test set", corpus_source_ids=[source.source_id])
    assert catalog.get_golden_dataset(dataset.dataset_id) is None
    catalog.save_golden_dataset(dataset)
    assert catalog.get_golden_dataset(dataset.dataset_id).dataset_id == dataset.dataset_id
    assert len(catalog.list_golden_datasets()) == 1

    record = ExperimentRecord(experiment_id="exp_1", manifest=ExperimentManifest(dataset_id=dataset.dataset_id), status="completed", started_at=utc_now(), completed_at=utc_now(), corpus_chunk_count=1)
    catalog.save_experiment(record)
    assert catalog.get_experiment(record.experiment_id).experiment_id == record.experiment_id
    assert len(catalog.list_experiments()) == 1

    assert catalog.get_cached_embedding("checksum-a", "model-v1") is None
    catalog.save_cached_embedding("checksum-a", "model-v1", [1.0, 2.0, 3.0])
    assert catalog.get_cached_embedding("checksum-a", "model-v1") == [1.0, 2.0, 3.0]
    assert catalog.get_cached_embedding("checksum-a", "model-v2") is None, "cache must be keyed by model version too"

    trace = QueryTrace(trace_id="trace-1", original_query="q", rewritten_query="q", profile="hybrid", vector_store="faiss", dense_candidates=0, lexical_candidates=0, fused_candidates=0, final_context_chunks=0, retrieval_ms=1.0, generation_ms=1.0, total_ms=2.0, model="granite4.2:3b", embedding_model="hash-512", embedding_fallback=True)
    assert catalog.get_query_trace("trace-1") is None
    catalog.save_query_trace("trace-1", trace)
    assert catalog.get_query_trace("trace-1").trace_id == "trace-1"
    history = catalog.list_query_traces(limit=10)
    assert history[0][0] == "trace-1"


def test_sqlite_catalog_satisfies_the_full_contract(tmp_path) -> None:
    _run_catalog_contract(SqliteCatalog(tmp_path / "catalog.db"))


@pytest.mark.integration
def test_postgres_catalog_satisfies_the_same_contract() -> None:
    """Real (unmocked) round trip against the docker-compose pgvector service -
    proves SqliteCatalog and PostgresCatalog are genuinely interchangeable behind
    CatalogPort, not just individually correct. Cleans up its own rows afterward
    since the service persists data across test runs, unlike a fresh SQLite file."""

    catalog = _postgres_or_skip()
    try:
        _run_catalog_contract(catalog)
    finally:
        catalog.connection.execute("DELETE FROM investrag_sources WHERE source_id = 'src_1'")
        catalog.connection.execute("DELETE FROM investrag_chunks_meta WHERE chunk_id = 'chunk_1'")
        catalog.connection.execute("DELETE FROM investrag_golden_datasets WHERE dataset_id = 'ds_1'")
        catalog.connection.execute("DELETE FROM investrag_experiments WHERE experiment_id = 'exp_1'")
        catalog.connection.execute("DELETE FROM investrag_embedding_cache WHERE checksum = 'checksum-a'")
        catalog.connection.execute("DELETE FROM investrag_query_traces WHERE trace_id = 'trace-1'")


def test_create_catalog_picks_sqlite_by_default(tmp_path) -> None:
    catalog = create_catalog(tmp_path / "catalog.db", None)
    assert isinstance(catalog, SqliteCatalog)


@pytest.mark.integration
def test_create_catalog_picks_postgres_when_dsn_configured(tmp_path) -> None:
    try:
        catalog = create_catalog(tmp_path / "catalog.db", _PG_DSN)
    except AdapterUnavailable as exc:
        pytest.skip(f"pgvector service unreachable: {exc}")
    assert isinstance(catalog, PostgresCatalog)
