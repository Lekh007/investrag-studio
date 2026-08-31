from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from .domain import Chunk, ExperimentRecord, GoldenDataset, QueryTrace, SourceVersion


class CatalogPort(Protocol):
    """Shared metadata-store contract. `SqliteCatalog` is the offline/test default
    (design §20: the core suite stays network-free); `PostgresCatalog` (design §4)
    is the same interface behind a real DSN. Nothing outside this module knows or
    cares which backend is active."""

    def save_source(self, source: SourceVersion) -> None: ...
    def update_source_review(self, source: SourceVersion) -> None: ...
    def get_source(self, source_id: str) -> SourceVersion | None: ...
    def list_sources(self) -> list[SourceVersion]: ...
    def save_chunks(self, chunks: list[Chunk]) -> None: ...
    def list_chunks(self) -> list[Chunk]: ...
    def save_golden_dataset(self, dataset: GoldenDataset) -> None: ...
    def get_golden_dataset(self, dataset_id: str) -> GoldenDataset | None: ...
    def list_golden_datasets(self) -> list[GoldenDataset]: ...
    def save_experiment(self, experiment: ExperimentRecord) -> None: ...
    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None: ...
    def list_experiments(self) -> list[ExperimentRecord]: ...
    def get_cached_embedding(self, checksum: str, model_version: str) -> list[float] | None: ...
    def save_cached_embedding(self, checksum: str, model_version: str, vector: list[float]) -> None: ...
    def save_query_trace(self, trace_id: str, trace: QueryTrace) -> None: ...
    def get_query_trace(self, trace_id: str) -> QueryTrace | None: ...
    def list_query_traces(self, limit: int = 50) -> list[tuple[str, QueryTrace]]: ...


class SqliteCatalog:
    """Local, file-based metadata adapter. Default backend; always available offline."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
              source_id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS golden_datasets (
              dataset_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
              experiment_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding_cache (
              checksum TEXT NOT NULL, model_version TEXT NOT NULL, vector TEXT NOT NULL,
              PRIMARY KEY (checksum, model_version)
            );
            CREATE TABLE IF NOT EXISTS query_traces (
              trace_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def save_source(self, source: SourceVersion) -> None:
        # INSERT OR IGNORE, not an upsert: a source version is immutable (design §5.1),
        # so re-ingesting identical bytes must never rewrite the existing record.
        self.connection.execute(
            "INSERT OR IGNORE INTO sources(source_id,name,payload) VALUES(?,?,?)",
            (source.source_id, source.name, source.model_dump_json()),
        )
        self.connection.commit()

    def update_source_review(self, source: SourceVersion) -> None:
        """The one legitimate mutation of an existing version: an operator's review
        decision (design §15.6). Deliberately a separate method from `save_source`,
        so the append-only guarantee there stays literally true and every caller that
        intends to change a stored version has to say so."""

        self.connection.execute(
            "UPDATE sources SET payload = ? WHERE source_id = ?",
            (source.model_dump_json(), source.source_id),
        )
        self.connection.commit()

    def get_source(self, source_id: str) -> SourceVersion | None:
        row = self.connection.execute(
            "SELECT payload FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return SourceVersion.model_validate_json(row[0]) if row else None

    def list_sources(self) -> list[SourceVersion]:
        rows = self.connection.execute("SELECT payload FROM sources ORDER BY rowid DESC").fetchall()
        return [SourceVersion.model_validate_json(row[0]) for row in rows]

    def save_chunks(self, chunks: list[Chunk]) -> None:
        self.connection.executemany("INSERT OR REPLACE INTO chunks(chunk_id,source_id,payload) VALUES(?,?,?)", [(chunk.chunk_id, chunk.source_id, chunk.model_dump_json()) for chunk in chunks])
        self.connection.commit()

    def list_chunks(self) -> list[Chunk]:
        rows = self.connection.execute("SELECT payload FROM chunks ORDER BY rowid").fetchall()
        return [Chunk.model_validate_json(row[0]) for row in rows]

    def save_golden_dataset(self, dataset: GoldenDataset) -> None:
        self.connection.execute("INSERT OR REPLACE INTO golden_datasets(dataset_id,payload) VALUES(?,?)", (dataset.dataset_id, dataset.model_dump_json()))
        self.connection.commit()

    def get_golden_dataset(self, dataset_id: str) -> GoldenDataset | None:
        row = self.connection.execute("SELECT payload FROM golden_datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return GoldenDataset.model_validate_json(row[0]) if row else None

    def list_golden_datasets(self) -> list[GoldenDataset]:
        rows = self.connection.execute("SELECT payload FROM golden_datasets ORDER BY rowid DESC").fetchall()
        return [GoldenDataset.model_validate_json(row[0]) for row in rows]

    def save_experiment(self, experiment: ExperimentRecord) -> None:
        self.connection.execute("INSERT OR REPLACE INTO experiments(experiment_id,payload) VALUES(?,?)", (experiment.experiment_id, experiment.model_dump_json()))
        self.connection.commit()

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        row = self.connection.execute("SELECT payload FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        return ExperimentRecord.model_validate_json(row[0]) if row else None

    def list_experiments(self) -> list[ExperimentRecord]:
        rows = self.connection.execute("SELECT payload FROM experiments ORDER BY rowid DESC").fetchall()
        return [ExperimentRecord.model_validate_json(row[0]) for row in rows]

    def get_cached_embedding(self, checksum: str, model_version: str) -> list[float] | None:
        row = self.connection.execute(
            "SELECT vector FROM embedding_cache WHERE checksum = ? AND model_version = ?", (checksum, model_version)
        ).fetchone()
        if row is None:
            return None
        return [float(value) for value in row[0].split(",")]

    def save_cached_embedding(self, checksum: str, model_version: str, vector: list[float]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO embedding_cache(checksum,model_version,vector) VALUES(?,?,?)",
            (checksum, model_version, ",".join(repr(value) for value in vector)),
        )
        self.connection.commit()

    def save_query_trace(self, trace_id: str, trace: QueryTrace) -> None:
        self.connection.execute("INSERT OR REPLACE INTO query_traces(trace_id,payload) VALUES(?,?)", (trace_id, trace.model_dump_json()))
        self.connection.commit()

    def get_query_trace(self, trace_id: str) -> QueryTrace | None:
        row = self.connection.execute("SELECT payload FROM query_traces WHERE trace_id = ?", (trace_id,)).fetchone()
        return QueryTrace.model_validate_json(row[0]) if row else None

    def list_query_traces(self, limit: int = 50) -> list[tuple[str, QueryTrace]]:
        rows = self.connection.execute("SELECT trace_id, payload FROM query_traces ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [(row[0], QueryTrace.model_validate_json(row[1])) for row in rows]


class PostgresCatalog:
    """Design §4's PostgreSQL metadata store. Same JSONB-blob schema shape as
    `SqliteCatalog` (idempotent `CREATE TABLE IF NOT EXISTS`, matching the rest of
    this project's adapters - Chroma/Qdrant/Milvus/pgvector all use the same
    ensure-schema-on-first-use pattern rather than a separate migration tool for a
    schema this simple)."""

    def __init__(self, dsn: str) -> None:
        from .vector_adapters import AdapterUnavailable

        try:
            import psycopg

            self.connection = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
        except Exception as exc:
            raise AdapterUnavailable(f"Postgres catalog unavailable: {exc}") from exc
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS investrag_sources (
              source_id TEXT PRIMARY KEY, name TEXT NOT NULL, payload JSONB NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS investrag_chunks_meta (
              chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, payload JSONB NOT NULL,
              seq BIGSERIAL
            );
            CREATE TABLE IF NOT EXISTS investrag_golden_datasets (
              dataset_id TEXT PRIMARY KEY, payload JSONB NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS investrag_experiments (
              experiment_id TEXT PRIMARY KEY, payload JSONB NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS investrag_embedding_cache (
              checksum TEXT NOT NULL, model_version TEXT NOT NULL, vector DOUBLE PRECISION[] NOT NULL,
              PRIMARY KEY (checksum, model_version)
            );
            CREATE TABLE IF NOT EXISTS investrag_query_traces (
              trace_id TEXT PRIMARY KEY, payload JSONB NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(), seq BIGSERIAL
            );
            """
        )

    def save_source(self, source: SourceVersion) -> None:
        # DO NOTHING, not DO UPDATE: see `SqliteCatalog.save_source`.
        self.connection.execute(
            "INSERT INTO investrag_sources(source_id,name,payload) VALUES(%s,%s,%s) ON CONFLICT (source_id) DO NOTHING",
            (source.source_id, source.name, source.model_dump_json()),
        )

    def update_source_review(self, source: SourceVersion) -> None:
        self.connection.execute(
            "UPDATE investrag_sources SET payload = %s WHERE source_id = %s",
            (source.model_dump_json(), source.source_id),
        )

    def get_source(self, source_id: str) -> SourceVersion | None:
        row = self.connection.execute("SELECT payload FROM investrag_sources WHERE source_id = %s", (source_id,)).fetchone()
        return SourceVersion.model_validate(row[0]) if row else None

    def list_sources(self) -> list[SourceVersion]:
        rows = self.connection.execute("SELECT payload FROM investrag_sources ORDER BY created_at DESC").fetchall()
        return [SourceVersion.model_validate(row[0]) for row in rows]

    def save_chunks(self, chunks: list[Chunk]) -> None:
        with self.connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO investrag_chunks_meta(chunk_id,source_id,payload) VALUES(%s,%s,%s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET payload = EXCLUDED.payload",
                [(chunk.chunk_id, chunk.source_id, chunk.model_dump_json()) for chunk in chunks],
            )

    def list_chunks(self) -> list[Chunk]:
        rows = self.connection.execute("SELECT payload FROM investrag_chunks_meta ORDER BY seq").fetchall()
        return [Chunk.model_validate(row[0]) for row in rows]

    def save_golden_dataset(self, dataset: GoldenDataset) -> None:
        self.connection.execute(
            "INSERT INTO investrag_golden_datasets(dataset_id,payload) VALUES(%s,%s) "
            "ON CONFLICT (dataset_id) DO UPDATE SET payload = EXCLUDED.payload",
            (dataset.dataset_id, dataset.model_dump_json()),
        )

    def get_golden_dataset(self, dataset_id: str) -> GoldenDataset | None:
        row = self.connection.execute("SELECT payload FROM investrag_golden_datasets WHERE dataset_id = %s", (dataset_id,)).fetchone()
        return GoldenDataset.model_validate(row[0]) if row else None

    def list_golden_datasets(self) -> list[GoldenDataset]:
        rows = self.connection.execute("SELECT payload FROM investrag_golden_datasets ORDER BY created_at DESC").fetchall()
        return [GoldenDataset.model_validate(row[0]) for row in rows]

    def save_experiment(self, experiment: ExperimentRecord) -> None:
        self.connection.execute(
            "INSERT INTO investrag_experiments(experiment_id,payload) VALUES(%s,%s) "
            "ON CONFLICT (experiment_id) DO UPDATE SET payload = EXCLUDED.payload",
            (experiment.experiment_id, experiment.model_dump_json()),
        )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        row = self.connection.execute("SELECT payload FROM investrag_experiments WHERE experiment_id = %s", (experiment_id,)).fetchone()
        return ExperimentRecord.model_validate(row[0]) if row else None

    def list_experiments(self) -> list[ExperimentRecord]:
        rows = self.connection.execute("SELECT payload FROM investrag_experiments ORDER BY created_at DESC").fetchall()
        return [ExperimentRecord.model_validate(row[0]) for row in rows]

    def get_cached_embedding(self, checksum: str, model_version: str) -> list[float] | None:
        row = self.connection.execute(
            "SELECT vector FROM investrag_embedding_cache WHERE checksum = %s AND model_version = %s", (checksum, model_version)
        ).fetchone()
        return [float(value) for value in row[0]] if row else None

    def save_cached_embedding(self, checksum: str, model_version: str, vector: list[float]) -> None:
        self.connection.execute(
            "INSERT INTO investrag_embedding_cache(checksum,model_version,vector) VALUES(%s,%s,%s) "
            "ON CONFLICT (checksum,model_version) DO UPDATE SET vector = EXCLUDED.vector",
            (checksum, model_version, vector),
        )

    def save_query_trace(self, trace_id: str, trace: QueryTrace) -> None:
        self.connection.execute(
            "INSERT INTO investrag_query_traces(trace_id,payload) VALUES(%s,%s) "
            "ON CONFLICT (trace_id) DO UPDATE SET payload = EXCLUDED.payload",
            (trace_id, trace.model_dump_json()),
        )

    def get_query_trace(self, trace_id: str) -> QueryTrace | None:
        row = self.connection.execute("SELECT payload FROM investrag_query_traces WHERE trace_id = %s", (trace_id,)).fetchone()
        return QueryTrace.model_validate(row[0]) if row else None

    def list_query_traces(self, limit: int = 50) -> list[tuple[str, QueryTrace]]:
        rows = self.connection.execute("SELECT trace_id, payload FROM investrag_query_traces ORDER BY seq DESC LIMIT %s", (limit,)).fetchall()
        return [(row[0], QueryTrace.model_validate(row[1])) for row in rows]


def create_catalog(sqlite_path: Path, postgres_dsn: str | None) -> CatalogPort:
    """Design §4/§20: Postgres when a DSN is configured, SQLite otherwise - the
    same port either way, and the default keeps the core test suite offline."""

    if postgres_dsn:
        return PostgresCatalog(postgres_dsn)
    return SqliteCatalog(sqlite_path)


# Backward-compatible alias - existing call sites and tests reference `Catalog`.
Catalog = SqliteCatalog
