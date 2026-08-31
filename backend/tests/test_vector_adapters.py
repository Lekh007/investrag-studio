from pathlib import Path

import numpy as np
import pytest

from investrag.domain import Chunk
from investrag.vector_adapters import (
    AdapterUnavailable,
    ChromaAdapter,
    MilvusAdapter,
    QdrantAdapter,
    adapter_capabilities,
)


def _fixture() -> tuple[Chunk, np.ndarray]:
    chunk = Chunk(
        chunk_id="chunk-a",
        source_id="source-a",
        profile="structure-aware",
        text="Revenue increased 12 percent.",
        element_ids=["element-a"],
        parent_id="parent-a",
        metadata={"page": 2, "unit": "%"},
    )
    return chunk, np.asarray([[1.0, 0.0]], dtype=np.float32)


def test_chroma_round_trips_full_chunk_provenance(tmp_path: Path) -> None:
    chunk, vector = _fixture()
    adapter = ChromaAdapter(tmp_path / "chroma")
    adapter.upsert([chunk], vector)

    records = adapter.search(vector[0], k=1)

    assert adapter.count == 1
    assert records[0][0] == chunk
    assert records[0][1] == pytest.approx(1.0)


def test_qdrant_round_trips_full_chunk_provenance_and_filter(tmp_path: Path) -> None:
    chunk, vector = _fixture()
    adapter = QdrantAdapter(tmp_path / "qdrant")
    adapter.upsert([chunk], vector)

    assert adapter.search(vector[0], k=1, source_ids=["missing"]) == []
    records = adapter.search(vector[0], k=1, source_ids=[chunk.source_id])
    assert adapter.count == 1
    assert records[0][0] == chunk


def test_milvus_round_trips_full_chunk_provenance_and_filter(tmp_path: Path) -> None:
    """Real (unmocked) round trip. milvus-lite's own PyPI-declared extra excludes
    win32 (a stale marker - the wheel itself works on Windows, verified 2026-08-30),
    so this project depends on milvus-lite directly rather than through
    pymilvus[milvus_lite]. Skips only if milvus-lite is genuinely absent."""

    capabilities = {row["name"]: row for row in adapter_capabilities(tmp_path)}
    if not capabilities["milvus"]["available"]:
        pytest.skip(f"milvus unavailable: {capabilities['milvus']['description']}")
    chunk, vector = _fixture()
    adapter = MilvusAdapter(tmp_path / "milvus")
    adapter.upsert([chunk], vector)

    assert adapter.count == 1
    assert adapter.search(vector[0], k=1, source_ids=["missing"]) == []
    records = adapter.search(vector[0], k=1, source_ids=[chunk.source_id])
    assert records[0][0] == chunk
    assert adapter.all_records() == [chunk]


_PGVECTOR_TEST_DSN = "postgresql://investrag:investrag@127.0.0.1:5433/investrag"


@pytest.mark.integration
def test_pgvector_round_trips_full_chunk_provenance_and_filter(tmp_path: Path) -> None:
    """Real (unmocked) round trip against the docker-compose pgvector service
    (design's own documented `docker compose up -d pgvector`). This adapter was
    coded but never actually exercised - proven live 2026-08-30: real CREATE
    EXTENSION, real vector column, real cosine search and source_id filter."""

    from investrag.vector_adapters import PgVectorAdapter

    try:
        adapter = PgVectorAdapter(_PGVECTOR_TEST_DSN, table_name="investrag_pytest")
    except AdapterUnavailable as exc:
        pytest.skip(f"pgvector service unreachable: {exc}")

    try:
        chunk, vector = _fixture()
        adapter.upsert([chunk], vector)

        assert adapter.count == 1
        assert adapter.search(vector[0], k=1, source_ids=["missing"]) == []
        records = adapter.search(vector[0], k=1, source_ids=[chunk.source_id])
        assert records[0][0] == chunk
        assert records[0][1] == pytest.approx(1.0)
        assert adapter.all_records() == [chunk]

        capabilities = {row["name"]: row for row in adapter_capabilities(tmp_path, _PGVECTOR_TEST_DSN)}
        assert capabilities["pgvector"]["available"] is True
    finally:
        adapter.connection.execute("DROP TABLE IF EXISTS investrag_pytest")


@pytest.mark.integration
def test_weaviate_round_trips_full_chunk_provenance_and_filter(tmp_path: Path) -> None:
    """Real (unmocked) round trip against `docker compose --profile weaviate up -d
    weaviate`. Weaviate is the only local store that genuinely needs Docker in this
    project - Qdrant/Milvus both run embedded, verified on Windows 2026-08-30."""

    from investrag.vector_adapters import WeaviateAdapter

    try:
        adapter = WeaviateAdapter("127.0.0.1", 8090, 50052, collection_name="InvestragPytest")
    except AdapterUnavailable as exc:
        pytest.skip(f"weaviate service unreachable: {exc}")

    try:
        chunk, vector = _fixture()
        adapter.upsert([chunk], vector)

        assert adapter.count == 1
        assert adapter.search(vector[0], k=1, source_ids=["missing"]) == []
        records = adapter.search(vector[0], k=1, source_ids=[chunk.source_id])
        assert records[0][0] == chunk
        assert records[0][1] == pytest.approx(1.0, abs=1e-4)
        assert adapter.all_records() == [chunk]

        capabilities = {row["name"]: row for row in adapter_capabilities(tmp_path, weaviate_host="127.0.0.1")}
        assert capabilities["weaviate"]["available"] is True
        assert capabilities["weaviate"]["mode"] == "local"
    finally:
        adapter.client.collections.delete("InvestragPytest")
        adapter.close()


def test_pinecone_reports_unavailable_without_an_api_key(tmp_path: Path) -> None:
    """The only thing honestly testable without a live Pinecone account: the
    capability-gating path. The adapter's actual upsert/query behaviour is
    implemented against Pinecone's documented SDK v9 API but is not exercised here -
    see PineconeAdapter's docstring for that disclosure."""

    capabilities = {row["name"]: row for row in adapter_capabilities(tmp_path)}
    assert capabilities["pinecone"]["implemented"] is True
    assert capabilities["pinecone"]["available"] is False
    assert "PINECONE_API_KEY" in capabilities["pinecone"]["description"]


def test_pinecone_adapter_construction_requires_a_key(tmp_path: Path) -> None:
    from investrag.vector_adapters import VectorStoreRegistry
    from investrag.vectorstore import FaissVectorStore

    registry = VectorStoreRegistry(tmp_path, FaissVectorStore(tmp_path / "faiss"))
    with pytest.raises(AdapterUnavailable):
        registry.get("pinecone")


def test_capabilities_do_not_claim_milvus_ready_without_lite_runtime(tmp_path: Path) -> None:
    capabilities = {row["name"]: row for row in adapter_capabilities(tmp_path)}
    assert capabilities["faiss"]["available"] is True
    assert capabilities["chroma"]["available"] is True
    assert capabilities["qdrant"]["available"] is True
    assert capabilities["pgvector"]["implemented"] is True
    assert capabilities["pgvector"]["available"] is False
    if not capabilities["milvus"]["available"]:
        with pytest.raises(AdapterUnavailable):
            MilvusAdapter(tmp_path / "milvus")


def test_milvus_uses_stable_non_overlapping_ids_across_publication_batches() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.batches = []

        def has_collection(self, _name: str) -> bool:
            return True

        def upsert(self, *, collection_name: str, data):
            self.batches.append((collection_name, data))

    adapter = object.__new__(MilvusAdapter)
    adapter.client = FakeClient()
    adapter.collection_name = "investrag"
    first, vector = _fixture()
    second = first.model_copy(update={"chunk_id": "chunk-b", "text": "Margin improved."})
    adapter.upsert([first], vector)
    adapter.upsert([second], vector)
    adapter.upsert([first], vector)
    ids = [batch[1][0]["id"] for batch in adapter.client.batches]
    assert ids[0] != ids[1]
    assert ids[0] == ids[2]
