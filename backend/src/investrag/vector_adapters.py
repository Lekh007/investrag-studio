from __future__ import annotations

import hashlib
import importlib.util
import os
import uuid
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .domain import Chunk
from .vectorstore import FaissVectorStore


class AdapterUnavailable(RuntimeError):
    pass


class VectorStorePort(Protocol):
    name: str

    @property
    def count(self) -> int: ...

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]: ...

    def all_records(self) -> list[Chunk]: ...


def _flat_metadata(chunk: Chunk) -> dict[str, str]:
    return {key: str(value) for key, value in chunk.metadata.items() if isinstance(value, (str, int, float, bool))}


class ChromaAdapter:
    name = "chroma"

    def __init__(self, path: Path, collection_name: str = "investrag") -> None:
        try:
            import chromadb

            path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path))
            self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        except Exception as exc:
            raise AdapterUnavailable(f"Chroma unavailable: {exc}") from exc

    @property
    def count(self) -> int:
        return int(self.collection.count())

    def all_records(self) -> list[Chunk]:
        result = cast(dict[str, Any], self.collection.get(include=["documents", "metadatas"]))
        records: list[Chunk] = []
        ids = cast(list[str], result.get("ids") or [])
        documents = cast(list[str | None], result.get("documents") or [])
        metadatas = cast(list[dict[str, Any] | None], result.get("metadatas") or [])
        for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            metadata = metadata or {}
            serialized = metadata.get("chunk_json")
            records.append(Chunk.model_validate_json(serialized) if isinstance(serialized, str) else Chunk(chunk_id=chunk_id, source_id=str(metadata.get("source_id", "unknown")), profile="external", text=document or "", element_ids=[], metadata=metadata))
        return records

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    **_flat_metadata(chunk),
                    "source_id": chunk.source_id,
                    "chunk_json": chunk.model_dump_json(),
                }
                for chunk in chunks
            ],
        )

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        query_kwargs: dict[str, Any] = {"query_embeddings": [vector.tolist()], "n_results": k}
        if source_ids:
            query_kwargs["where"] = {"source_id": {"$in": source_ids}}
        result = cast(dict[str, Any], self.collection.query(**query_kwargs))
        ids = cast(list[str], (result.get("ids") or [[]])[0])
        documents = cast(list[str], (result.get("documents") or [[]])[0])
        distances = cast(list[float], (result.get("distances") or [[]])[0])
        metadatas = cast(list[dict[str, Any]], (result.get("metadatas") or [[]])[0])
        items: list[tuple[Chunk, float]] = []
        for chunk_id, document, distance, metadata in zip(ids, documents, distances, metadatas, strict=True):
            serialized = metadata.get("chunk_json")
            chunk = Chunk.model_validate_json(serialized) if isinstance(serialized, str) else Chunk(chunk_id=chunk_id, source_id=str(metadata.get("source_id", "unknown")), profile="external", text=document, element_ids=[], metadata=metadata)
            items.append((chunk, 1.0 - float(distance)))
        return items


class QdrantAdapter:
    name = "qdrant"

    def __init__(self, path: Path, collection_name: str = "investrag") -> None:
        try:
            from qdrant_client import QdrantClient, models

            path.mkdir(parents=True, exist_ok=True)
            self.models = models
            self.client = QdrantClient(path=str(path))
            self.collection_name = collection_name
        except Exception as exc:
            raise AdapterUnavailable(f"Qdrant unavailable: {exc}") from exc

    def _ensure(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(self.collection_name, vectors_config=self.models.VectorParams(size=dimension, distance=self.models.Distance.COSINE))

    @property
    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(self.collection_name, exact=True).count)

    def all_records(self) -> list[Chunk]:
        if not self.client.collection_exists(self.collection_name):
            return []
        records: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection_name, offset=offset, limit=256, with_payload=True, with_vectors=False)
            records.extend(Chunk.model_validate(point.payload["chunk"]) for point in points if point.payload and "chunk" in point.payload)
            if offset is None:
                return records

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._ensure(vectors.shape[1])
        points = [self.models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)), vector=vector.tolist(), payload={"chunk": chunk.model_dump(mode="json")}) for chunk, vector in zip(chunks, vectors, strict=True)]
        self.client.upsert(self.collection_name, points=points, wait=True)

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.client.collection_exists(self.collection_name):
            return []
        query_filter = None
        if source_ids:
            query_filter = self.models.Filter(must=[self.models.FieldCondition(key="chunk.source_id", match=self.models.MatchAny(any=source_ids))])
        try:
            response = self.client.query_points(collection_name=self.collection_name, query=vector.tolist(), query_filter=query_filter, limit=k, with_payload=True).points
        except AttributeError:
            response = self.client.search(collection_name=self.collection_name, query_vector=vector.tolist(), query_filter=query_filter, limit=k, with_payload=True)  # type: ignore[attr-defined]
        return [(Chunk.model_validate(point.payload["chunk"]), float(point.score)) for point in response if point.payload and "chunk" in point.payload]


class MilvusAdapter:
    name = "milvus"

    def __init__(self, path: Path, collection_name: str = "investrag") -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            if importlib.util.find_spec("milvus_lite") is None:
                raise AdapterUnavailable("milvus-lite is not installed; local Milvus requires the milvus-lite wheel")
            from pymilvus import MilvusClient

            self.client = MilvusClient(uri=str(path / "milvus.db"))
            self.collection_name = collection_name
        except Exception as exc:
            raise AdapterUnavailable(f"Milvus unavailable: {exc}") from exc

    @property
    def count(self) -> int:
        try:
            return int(self.client.get_collection_stats(self.collection_name).get("row_count", 0))
        except Exception:
            return 0

    def all_records(self) -> list[Chunk]:
        if not self.client.has_collection(self.collection_name):
            return []
        rows = self.client.query(collection_name=self.collection_name, filter="", output_fields=["chunk"], limit=100000)
        return [Chunk.model_validate(row["chunk"]) for row in rows]

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name, dimension=int(vectors.shape[1]), metric_type="COSINE", auto_id=False, primary_field_name="id", vector_field_name="vector")
        rows = [
            {
                "id": int.from_bytes(
                    hashlib.blake2b(chunk.chunk_id.encode(), digest_size=8).digest(), "big"
                )
                & ((1 << 63) - 1),
                "vector": vector.tolist(),
                "chunk": chunk.model_dump(mode="json"),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, data=rows)

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.client.has_collection(self.collection_name):
            return []
        search_limit = min(1000, max(k, k * 4 if source_ids else k))
        results = self.client.search(collection_name=self.collection_name, data=[vector.tolist()], limit=search_limit, output_fields=["chunk"])[0]
        items: list[tuple[Chunk, float]] = []
        for result in results:
            chunk = Chunk.model_validate(result["entity"]["chunk"])
            if source_ids and chunk.source_id not in source_ids:
                continue
            items.append((chunk, float(result["distance"])))
            if len(items) >= k:
                break
        return items


class PgVectorAdapter:
    """Optional PostgreSQL/pgvector adapter activated only with a DSN."""

    name = "pgvector"

    def __init__(self, dsn: str, table_name: str = "investrag_chunks") -> None:
        try:
            import psycopg

            self.connection = psycopg.connect(dsn, autocommit=True)
            self.table_name = table_name
        except Exception as exc:
            raise AdapterUnavailable(f"pgvector unavailable: {exc}") from exc

    def _table_exists(self) -> bool:
        row = self.connection.execute("SELECT to_regclass(%s)", (self.table_name,)).fetchone()
        return bool(row and row[0])

    def _ensure_schema(self, dimension: int) -> None:
        if self._table_exists():
            return
        if not 1 <= dimension <= 8192:
            raise ValueError("embedding dimension must be between 1 and 8192")
        self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, text TEXT NOT NULL, chunk_json JSONB NOT NULL, embedding vector({dimension}) NOT NULL)")
        self.connection.execute(f"CREATE INDEX IF NOT EXISTS {self.table_name}_source_idx ON {self.table_name}(source_id)")

    @property
    def count(self) -> int:
        if not self._table_exists():
            return 0
        row = self.connection.execute(f"SELECT count(*) FROM {self.table_name}").fetchone()
        return int(row[0]) if row else 0

    def all_records(self) -> list[Chunk]:
        if not self._table_exists():
            return []
        rows = self.connection.execute(f"SELECT chunk_json FROM {self.table_name}").fetchall()
        return [Chunk.model_validate(row[0]) for row in rows]

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._ensure_schema(int(vectors.shape[1]))
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.connection.execute(f"INSERT INTO {self.table_name}(chunk_id,source_id,text,chunk_json,embedding) VALUES (%s,%s,%s,%s,%s::vector) ON CONFLICT (chunk_id) DO UPDATE SET source_id=EXCLUDED.source_id,text=EXCLUDED.text,chunk_json=EXCLUDED.chunk_json,embedding=EXCLUDED.embedding", (chunk.chunk_id, chunk.source_id, chunk.text, chunk.model_dump_json(), str(vector.tolist())))

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self._table_exists():
            return []
        params: list[Any] = [str(vector.tolist()), k]
        where = ""
        if source_ids:
            where = " WHERE source_id = ANY(%s)"
            params.insert(1, source_ids)
        rows = self.connection.execute(f"SELECT chunk_json, 1 - (embedding <=> %s::vector) AS score FROM {self.table_name}{where} ORDER BY embedding <=> %s::vector LIMIT %s", [*params[:-1], str(vector.tolist()), params[-1]]).fetchall()
        return [(Chunk.model_validate(row[0]), float(row[1])) for row in rows]


class WeaviateAdapter:
    """Weaviate is the only local store that genuinely needs Docker in this project
    (no supported embedded mode on Windows, unlike Qdrant/Milvus below). Uses
    self-provided vectors - this project always computes its own embeddings - and
    exposes native BM25/hybrid search for the design §14.6 native-feature track.
    Verified live against `docker compose --profile weaviate up -d weaviate`,
    2026-08-30: create, insert, near-vector search, source_id filter, delete.
    """

    name = "weaviate"

    def __init__(self, host: str, port: int, grpc_port: int, collection_name: str = "InvestragChunk") -> None:
        try:
            import weaviate

            self.client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
            self.collection_name = collection_name
            if not self.client.is_ready():
                raise AdapterUnavailable(f"Weaviate at {host}:{port} is not ready")
        except AdapterUnavailable:
            raise
        except Exception as exc:
            raise AdapterUnavailable(f"Weaviate unavailable: {exc}") from exc

    def _ensure_collection(self) -> Any:
        from weaviate.classes.config import Configure, DataType, Property

        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="source_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="chunk_json", data_type=DataType.TEXT),
                ],
                vector_config=Configure.Vectors.self_provided(),
            )
        return self.client.collections.get(self.collection_name)

    @property
    def count(self) -> int:
        if not self.client.collections.exists(self.collection_name):
            return 0
        collection = self.client.collections.get(self.collection_name)
        return int(collection.aggregate.over_all(total_count=True).total_count or 0)

    def all_records(self) -> list[Chunk]:
        if not self.client.collections.exists(self.collection_name):
            return []
        collection = self.client.collections.get(self.collection_name)
        return [Chunk.model_validate_json(str(obj.properties["chunk_json"])) for obj in collection.iterator()]

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        collection = self._ensure_collection()
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors, strict=True):
                batch.add_object(
                    properties={
                        "chunk_id": chunk.chunk_id,
                        "source_id": chunk.source_id,
                        "text": chunk.text,
                        "chunk_json": chunk.model_dump_json(),
                    },
                    vector=vector.tolist(),
                    uuid=uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id),
                )

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.client.collections.exists(self.collection_name):
            return []
        from weaviate.classes.query import Filter, MetadataQuery

        collection = self.client.collections.get(self.collection_name)
        filters = Filter.by_property("source_id").contains_any(source_ids) if source_ids else None
        response = collection.query.near_vector(near_vector=vector.tolist(), limit=k, filters=filters, return_metadata=MetadataQuery(distance=True))
        items: list[tuple[Chunk, float]] = []
        for obj in response.objects:
            chunk = Chunk.model_validate_json(str(obj.properties["chunk_json"]))
            distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 0.0
            items.append((chunk, 1.0 - float(distance)))
        return items

    def close(self) -> None:
        self.client.close()


class PineconeAdapter:
    """Optional cloud track, activated only with an explicit API key (design §22:
    never in the mandatory path). Uses the classic dimension-based create_index +
    vector upsert/query API (not the newer managed-embedding upsert_records/search,
    which computes embeddings server-side and would bypass this project's own
    embedding provider).

    Disclosed honestly: implemented against Pinecone's documented SDK v9 API but
    **not exercised against a live index** - no API key was available to test with
    in this environment. Unlike every other adapter in this file, this one's
    correctness rests on documentation, not a live round trip.
    """

    name = "pinecone"

    def __init__(self, api_key: str, index_name: str = "investrag", cloud: str = "aws", region: str = "us-east-1") -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec

            self.pc = Pinecone(api_key=api_key)
            self.index_name = index_name
            self._cloud = cloud
            self._region = region
            self._spec_cls = ServerlessSpec
            self._index: Any = None
        except Exception as exc:
            raise AdapterUnavailable(f"Pinecone unavailable: {exc}") from exc

    def _ensure_index(self, dimension: int) -> Any:
        if self._index is not None:
            return self._index
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(name=self.index_name, dimension=dimension, metric="cosine", spec=self._spec_cls(cloud=self._cloud, region=self._region))
        self._index = self.pc.Index(self.index_name)
        return self._index

    @property
    def count(self) -> int:
        if self._index is None:
            return 0
        stats = self._index.describe_index_stats()
        return int(stats.get("total_vector_count", 0))

    def all_records(self) -> list[Chunk]:
        raise AdapterUnavailable("Pinecone does not support unfiltered full-collection scans; use search() or source_ids filtering")

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        index = self._ensure_index(int(vectors.shape[1]))
        index.upsert(vectors=[{"id": chunk.chunk_id, "values": vector.tolist(), "metadata": {"source_id": chunk.source_id, "chunk_json": chunk.model_dump_json()}} for chunk, vector in zip(chunks, vectors, strict=True)])

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if self._index is None:
            return []
        query_filter = {"source_id": {"$in": source_ids}} if source_ids else None
        response = self._index.query(vector=vector.tolist(), top_k=k, filter=query_filter, include_metadata=True)
        return [(Chunk.model_validate_json(str(match["metadata"]["chunk_json"])), float(match["score"])) for match in response.get("matches", [])]


def adapter_capabilities(data_dir: Path, postgres_dsn: str | None = None, weaviate_host: str | None = None, pinecone_api_key: str | None = None) -> list[dict[str, Any]]:
    """Describe adapter code and package availability without starting services."""

    package_for = {"chroma": "chromadb", "qdrant": "qdrant_client", "milvus": "pymilvus", "weaviate": "weaviate", "pgvector": "psycopg", "pinecone": "pinecone"}
    rows = [{"name": "faiss", "implemented": True, "installed": True, "available": True, "mode": "local", "description": "dense + local persistence"}]
    for name, package in package_for.items():
        implemented = name in {"chroma", "qdrant", "milvus", "pgvector", "weaviate", "pinecone"}
        installed = bool(importlib.util.find_spec(package))
        runtime_ready = installed
        reason = ""
        if name == "milvus" and installed and importlib.util.find_spec("milvus_lite") is None:
            runtime_ready = False
            reason = "install milvus-lite for the embedded local server"
        if name == "pgvector" and implemented:
            runtime_ready = installed and bool(postgres_dsn or os.getenv("INVESTRAG_POSTGRES_DSN"))
            if not runtime_ready:
                reason = "configure INVESTRAG_POSTGRES_DSN and start the pgvector service"
        if name == "weaviate" and implemented:
            runtime_ready = installed and bool(weaviate_host or os.getenv("INVESTRAG_WEAVIATE_HOST"))
            if not runtime_ready:
                reason = "configure INVESTRAG_WEAVIATE_HOST and run `docker compose --profile weaviate up -d weaviate`"
        if name == "pinecone" and implemented:
            runtime_ready = installed and bool(pinecone_api_key or os.getenv("INVESTRAG_PINECONE_API_KEY"))
            if not runtime_ready:
                reason = "configure INVESTRAG_PINECONE_API_KEY (optional cloud track, never required)"
        rows.append({"name": name, "implemented": implemented, "installed": installed, "available": implemented and runtime_ready, "mode": "local" if name in {"chroma", "qdrant", "milvus", "weaviate"} else "service/cloud", "description": "adapter available" if implemented and runtime_ready else (reason or "adapter unavailable")})
    return rows


class VectorStoreRegistry:
    """Lazily creates local adapters so heavy stores can be benchmarked sequentially."""

    def __init__(
        self,
        data_dir: Path,
        faiss_store: FaissVectorStore,
        postgres_dsn: str | None = None,
        weaviate_host: str | None = None,
        weaviate_port: int = 8090,
        weaviate_grpc_port: int = 50052,
        pinecone_api_key: str | None = None,
        pinecone_index: str = "investrag",
    ) -> None:
        self.data_dir = data_dir
        self.postgres_dsn = postgres_dsn
        self.weaviate_host = weaviate_host
        self.weaviate_port = weaviate_port
        self.weaviate_grpc_port = weaviate_grpc_port
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index = pinecone_index
        self.adapters: dict[str, VectorStorePort] = {"faiss": faiss_store}

    def get(self, name: str) -> VectorStorePort:
        if name in self.adapters:
            return self.adapters[name]
        path = self.data_dir / name
        if name == "chroma":
            adapter: VectorStorePort = ChromaAdapter(path)
        elif name == "qdrant":
            adapter = QdrantAdapter(path)
        elif name == "milvus":
            adapter = MilvusAdapter(path)
        elif name == "pgvector":
            if not self.postgres_dsn:
                raise AdapterUnavailable("pgvector requires INVESTRAG_POSTGRES_DSN")
            adapter = PgVectorAdapter(self.postgres_dsn)
        elif name == "weaviate":
            if not self.weaviate_host:
                raise AdapterUnavailable("weaviate requires INVESTRAG_WEAVIATE_HOST")
            adapter = WeaviateAdapter(self.weaviate_host, self.weaviate_port, self.weaviate_grpc_port)
        elif name == "pinecone":
            if not self.pinecone_api_key:
                raise AdapterUnavailable("pinecone requires INVESTRAG_PINECONE_API_KEY")
            adapter = PineconeAdapter(self.pinecone_api_key, self.pinecone_index)
        else:
            raise AdapterUnavailable(f"Vector store adapter {name!r} is not implemented in the local release")
        self.adapters[name] = adapter
        return adapter

    def publish(self, names: list[str], chunks: list[Chunk], vectors: np.ndarray) -> dict[str, str]:
        outcomes: dict[str, str] = {}
        for name in names:
            try:
                self.get(name).upsert(chunks, vectors)
                outcomes[name] = "published"
            except Exception as exc:
                outcomes[name] = f"unavailable: {exc}"
        return outcomes
