"""Per-database integration tests (design §20: "integration tests for each local vector
database, tagged so heavy services run sequentially").

Every test here is marked ``integration`` and is therefore **deselected by the default
``pytest`` invocation** (see ``addopts`` in ``pyproject.toml``), which is what keeps
design §20's last paragraph true: the continuous suite needs no network, no GPU and no
database. Run these deliberately::

    uv run pytest -m integration

What separates this file from ``test_vector_adapters.py`` is the layer under test.
That file exercises an adapter in isolation - construct, upsert, search, filter. This
one drives the **whole service** against each store in turn: a real
``InvestRAGService.ingest`` parses, chunks, embeds and *publishes* to the store, and a
real ``InvestRAGService.query`` retrieves back out of it. A store that round-trips
vectors but loses chunk provenance, or that the publication path never actually reaches,
passes there and fails here.

Sequential by construction, per §20 and §19:

* exactly one adapter is open at a time - each test constructs its own and closes it in
  a ``finally``, so Milvus and Weaviate are never resident together;
* every store gets a run-unique collection/table so a shared local service is never
  polluted across runs, and service-backed stores clean up after themselves;
* embeddings run in deterministic ``hash`` mode, so nothing here contends for the GPU.

**Never run this marker under ``pytest-xdist``.** Parallel workers would defeat the
one-store-at-a-time property this file exists to preserve.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from investrag.config import Settings
from investrag.domain import QueryRequest
from investrag.service import InvestRAGService
from investrag.vector_adapters import AdapterUnavailable, VectorStorePort

pytestmark = pytest.mark.integration

PGVECTOR_DSN = "postgresql://investrag:investrag@127.0.0.1:5433/investrag"
WEAVIATE_HOST = "127.0.0.1"
WEAVIATE_PORT = 8090
WEAVIATE_GRPC_PORT = 50052

_CORPUS = b"""# Q3 investment note

Revenue for the quarter was $412 million, up from $380 million in Q2.

Operating margin improved to 18.2 percent on lower logistics costs.
"""

_DECOY = b"""# Unrelated municipal filing

The bridge maintenance levy was renewed for a further five years.
"""


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@contextmanager
def _chroma(tmp_path: Path) -> Iterator[VectorStorePort]:
    from investrag.vector_adapters import ChromaAdapter

    yield ChromaAdapter(tmp_path / "chroma", collection_name=_unique("it"))


@contextmanager
def _qdrant(tmp_path: Path) -> Iterator[VectorStorePort]:
    from investrag.vector_adapters import QdrantAdapter

    yield QdrantAdapter(tmp_path / "qdrant", collection_name=_unique("it"))


@contextmanager
def _milvus(tmp_path: Path) -> Iterator[VectorStorePort]:
    from investrag.vector_adapters import MilvusAdapter

    yield MilvusAdapter(tmp_path / "milvus", collection_name=_unique("it"))


@contextmanager
def _pgvector(_tmp_path: Path) -> Iterator[VectorStorePort]:
    """Shared service, so the table name is run-unique and dropped afterwards."""

    from investrag.vector_adapters import PgVectorAdapter

    adapter = PgVectorAdapter(PGVECTOR_DSN, table_name=_unique("investrag_it"))
    try:
        yield adapter
    finally:
        adapter.connection.execute(f"DROP TABLE IF EXISTS {adapter.table_name}")


@contextmanager
def _weaviate(_tmp_path: Path) -> Iterator[VectorStorePort]:
    """Shared service, and Weaviate class names must be CamelCase, so the collection is
    run-unique and deleted afterwards."""

    from investrag.vector_adapters import WeaviateAdapter

    name = "InvestragIt" + uuid.uuid4().hex[:10]
    adapter = WeaviateAdapter(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT, collection_name=name)
    try:
        yield adapter
    finally:
        try:
            adapter.client.collections.delete(name)
        finally:
            adapter.client.close()


# Ordered light -> heavy so an interrupted run still covers the embedded stores, and so
# the two service-backed stores are the last things holding resources.
STORE_FACTORIES: dict[str, Callable[[Path], object]] = {
    "chroma": _chroma,
    "qdrant": _qdrant,
    "milvus": _milvus,
    "pgvector": _pgvector,
    "weaviate": _weaviate,
}


def _service(tmp_path: Path, store_name: str) -> InvestRAGService:
    return InvestRAGService(
        Settings(
            data_dir=tmp_path / "data",
            embedding_mode="hash",
            ollama_base_url="http://127.0.0.1:1",
            publish_vector_stores=store_name,
        )
    )


def _run_store_contract(service: InvestRAGService, store_name: str, adapter: VectorStorePort, tmp_path: Path) -> None:
    """The same body for every database. Anything asserted here is a property design
    §10 puts in the portable adapter contract, exercised through the service rather
    than against the adapter directly."""

    # Inject the run-unique adapter so the publication path in `service.ingest` reaches
    # *this* collection rather than the registry's shared default one.
    service.registry.adapters[store_name] = adapter

    note = tmp_path / "note.md"
    note.write_bytes(_CORPUS)
    decoy = tmp_path / "decoy.md"
    decoy.write_bytes(_DECOY)

    published = service.ingest(note, "note.md")
    other = service.ingest(decoy, "decoy.md")
    assert published.status == "ready" and published.queryable is True
    assert not any("unavailable" in warning for warning in published.warnings), (
        f"{store_name} publication reported a failure: {published.warnings}"
    )

    # 1. Publication actually landed in this store, not only in the always-on FAISS index.
    assert adapter.count >= published.chunk_count > 0

    # 2. Full chunk provenance survives the store round trip - the catalog's record and
    #    the store's record must be the same object, not merely the same text.
    catalog_chunks = {chunk.chunk_id: chunk for chunk in service.catalog.list_chunks()}
    for record in adapter.all_records():
        expected = catalog_chunks.get(record.chunk_id)
        if expected is None:
            continue
        assert record.source_id == expected.source_id
        assert record.text == expected.text
        assert record.element_ids == expected.element_ids
        assert record.metadata == expected.metadata

    # 3. Retrieval through the real query pipeline, reading from this store.
    response = service.query(
        QueryRequest(question="What was revenue for the quarter?", profile="dense", vector_store=store_name, top_k=5)
    )
    assert response.citations, f"{store_name} returned no citations through service.query"
    assert response.trace.vector_store == store_name
    assert any("412" in citation.excerpt for citation in response.citations)
    assert all(citation.source_name for citation in response.citations)

    # 4. Metadata filtering: restricting to the decoy source must exclude this evidence.
    filtered = service.query(
        QueryRequest(
            question="What was revenue for the quarter?",
            profile="dense",
            vector_store=store_name,
            top_k=5,
            source_ids=[other.source_id],
        )
    )
    assert all(citation.source_id != published.source_id for citation in filtered.citations), (
        f"{store_name} ignored the source_id filter"
    )


@pytest.mark.parametrize("store_name", list(STORE_FACTORIES))
def test_each_vector_database_satisfies_the_service_level_contract(store_name: str, tmp_path: Path) -> None:
    factory = STORE_FACTORIES[store_name]
    service = _service(tmp_path, store_name)
    try:
        with factory(tmp_path) as adapter:  # type: ignore[operator]
            _run_store_contract(service, store_name, adapter, tmp_path)
    except AdapterUnavailable as exc:
        pytest.skip(f"{store_name} unavailable in this environment: {exc}")


def test_faiss_satisfies_the_same_contract_as_the_service_default(tmp_path: Path) -> None:
    """FAISS is always present and is the store every ingestion publishes to, so it is
    the control: if this fails, the failure is in the pipeline, not in an adapter."""

    service = _service(tmp_path, "faiss")
    _run_store_contract(service, "faiss", service.store, tmp_path)


def test_a_store_that_is_down_degrades_instead_of_failing_the_ingestion(tmp_path: Path) -> None:
    """Design §17: "one database publication failure does not invalidate successful
    publications to other stores" and "a database outage removes that adapter from
    selection without crashing the application". Exercised against a real unreachable
    pgvector DSN rather than a mocked-out adapter."""

    service = InvestRAGService(
        Settings(
            data_dir=tmp_path / "data",
            embedding_mode="hash",
            ollama_base_url="http://127.0.0.1:1",
            publish_vector_stores="pgvector",
            postgres_dsn="postgresql://investrag:investrag@127.0.0.1:1/investrag",
        )
    )
    note = tmp_path / "note.md"
    note.write_bytes(_CORPUS)

    source = service.ingest(note, "note.md")

    assert source.status == "ready", "an unreachable optional store must not fail the ingestion"
    assert source.queryable is True
    assert any(warning.startswith("pgvector: unavailable") for warning in source.warnings), (
        f"the outage must be visible in the version's warnings, got {source.warnings}"
    )
    # FAISS still received the chunks, so the corpus remains queryable.
    assert service.query(
        QueryRequest(question="What was revenue for the quarter?", profile="dense", vector_store="faiss", top_k=5)
    ).citations
