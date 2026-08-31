import json

import numpy as np

from investrag.domain import Chunk
from investrag.embeddings import EmbeddingProvider
from investrag.retrieval import BM25, LangChainRetriever
from investrag.vectorstore import FaissVectorStore


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="a", source_id="source-a", profile="structure-aware", text="revenue increased", element_ids=["a"]),
        Chunk(chunk_id="b", source_id="source-b", profile="structure-aware", text="revenue declined", element_ids=["b"]),
    ]


def test_bm25_honors_source_filter() -> None:
    records = _chunks()
    results = BM25(records).search("revenue", source_ids=["source-b"])
    assert [record.source_id for record, _ in results] == ["source-b"]


def test_mmr_returns_requested_count(tmp_path) -> None:
    records = _chunks()
    vectors = EmbeddingProvider("unused", mode="hash", fallback_dimension=16).embed([record.text for record in records]).vectors
    store = FaissVectorStore(tmp_path)
    store.upsert(records, np.asarray(vectors, dtype=np.float32))
    retriever = LangChainRetriever(store, EmbeddingProvider("unused", mode="hash", fallback_dimension=16))
    assert len(retriever.search("revenue", "mmr", 2)) == 2


def test_rrf_counts_each_retriever_contribution_once() -> None:
    dense_only = Chunk(chunk_id="dense", source_id="source-a", profile="structure-aware", text="unrelated", element_ids=["a"])
    overlap = Chunk(chunk_id="overlap", source_id="source-b", profile="structure-aware", text="alpha evidence", element_ids=["b"])

    class FakeStore:
        name = "fake"
        count = 2

        def search(self, _vector, k=10, source_ids=None):
            return [(dense_only, 0.9), (overlap, 0.8)][:k]

        def all_records(self):
            return [dense_only, overlap]

        def upsert(self, _chunks, _vectors):
            return None

    run = LangChainRetriever(
        FakeStore(), EmbeddingProvider("unused", mode="hash", fallback_dimension=16)
    ).invoke("alpha", "hybrid", 2)
    by_id = {item.chunk.chunk_id: item.score for item in run.results}
    assert by_id["dense"] == 1 / 61
    assert by_id["overlap"] == 1 / 62 + 1 / 61
    assert run.results[0].chunk.chunk_id == "overlap"


def test_faiss_store_loads_persisted_numpy_fallback(tmp_path) -> None:
    records = _chunks()
    vectors = EmbeddingProvider("unused", mode="hash", fallback_dimension=16).embed(
        [record.text for record in records]
    ).vectors
    (tmp_path / "chunks.json").write_text(
        json.dumps([record.model_dump(mode="json") for record in records]), encoding="utf-8"
    )
    np.save(tmp_path / "chunks.npy", vectors)

    store = FaissVectorStore(tmp_path)
    assert store.count == 2
    assert store.dimension == 16
    assert store.search(vectors[0], k=1)[0][0].chunk_id == "a"


def test_multi_query_uses_deterministic_templates_without_an_answerer(tmp_path) -> None:
    records = _chunks()
    vectors = EmbeddingProvider("unused", mode="hash", fallback_dimension=16).embed([r.text for r in records]).vectors
    store = FaissVectorStore(tmp_path)
    store.upsert(records, np.asarray(vectors, dtype=np.float32))
    run = LangChainRetriever(store, EmbeddingProvider("unused", mode="hash", fallback_dimension=16)).invoke(
        "revenue", "multi-query", 2
    )
    variant_stage = run.stages[0]
    assert variant_stage["generation_mode"] == "extractive-fallback"
    assert variant_stage["variant_count"] == 3  # question + 2 deterministic templates


def test_multi_query_uses_llm_generated_variants_when_an_answerer_is_configured(
    monkeypatch, tmp_path
) -> None:
    import httpx

    from investrag.llm import LocalAnswerer

    def fake_post(*_args, **_kwargs):
        return httpx.Response(
            200,
            json={"response": "alternative phrasing one\nalternative phrasing two"},
            request=httpx.Request("POST", "http://x"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    records = _chunks()
    vectors = EmbeddingProvider("unused", mode="hash", fallback_dimension=16).embed([r.text for r in records]).vectors
    store = FaissVectorStore(tmp_path)
    store.upsert(records, np.asarray(vectors, dtype=np.float32))
    answerer = LocalAnswerer("http://127.0.0.1:11434", "granite4.2:3b")
    run = LangChainRetriever(store, EmbeddingProvider("unused", mode="hash", fallback_dimension=16), answerer=answerer).invoke(
        "revenue", "multi-query", 2
    )
    variant_stage = run.stages[0]
    assert variant_stage["generation_mode"] == "model"
    assert variant_stage["variant_count"] == 3  # question + 2 LLM-generated variants


def test_parent_profile_expands_a_child_hit_to_its_real_parent_chunk(tmp_path) -> None:
    from investrag.chunking import chunk_elements
    from investrag.domain import CanonicalElement

    elements = [
        CanonicalElement(element_id="h1", element_type="heading", text="Risk Factors"),
        CanonicalElement(element_id="p1", element_type="paragraph", text="Credit spreads widened materially across the loan portfolio this quarter."),
        CanonicalElement(element_id="p2", element_type="paragraph", text="Watchlist loans increased from four to seven during the period under review."),
    ]
    chunks = chunk_elements(elements, "src_parent", profile="parent-child")
    parent = next(c for c in chunks if c.metadata.get("role") == "parent")
    embeddings = EmbeddingProvider("unused", mode="hash", fallback_dimension=16)
    vectors = embeddings.embed([c.text for c in chunks]).vectors
    store = FaissVectorStore(tmp_path)
    store.upsert(chunks, vectors)

    run = LangChainRetriever(store, embeddings).invoke("credit spreads widened", "parent", 3)

    assert run.results, "the child chunk mentioning credit spreads must be found"
    top = run.results[0]
    assert top.chunk.chunk_id == parent.chunk_id, "the returned chunk must be the real parent, not the child hit"
    assert top.chunk.metadata.get("role") == "parent"
    assert "Watchlist loans" in top.chunk.text, "the parent must carry the FULL section, including sentences the child hit did not contain"


def test_parent_profile_falls_back_to_the_candidate_when_parent_id_is_not_a_stored_chunk(tmp_path) -> None:
    """structure-aware chunks point parent_id at a heading *element* id, never a
    chunk id in the store - this must not silently disappear the result."""

    records = _chunks()  # structure-aware-style chunks with no parent_id set
    embeddings = EmbeddingProvider("unused", mode="hash", fallback_dimension=16)
    vectors = embeddings.embed([r.text for r in records]).vectors
    store = FaissVectorStore(tmp_path)
    store.upsert(records, np.asarray(vectors, dtype=np.float32))

    run = LangChainRetriever(store, embeddings).invoke("revenue", "parent", 2)
    assert len(run.results) == 2
    assert {item.chunk.chunk_id for item in run.results} == {"a", "b"}
