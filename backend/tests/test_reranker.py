"""Real (unmocked) reranker coverage. mxbai-rerank-base-v2 is chosen over the design's
original bge-reranker-v2-m3 (absent from current reranker trending) and over
NVIDIA/Jina alternatives that require trust_remote_code=True - see reranker.py's
docstring for the full comparison. Model download happens lazily and is skipped if
unreachable, matching how the embedding provider already behaves offline."""

from __future__ import annotations

import pytest

from investrag.reranker import CrossEncoderReranker


def _reranker_or_skip() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker()
    if not reranker.available:
        pytest.skip(f"reranker model unavailable: {reranker._unavailable_reason}")
    return reranker


def test_reranker_ranks_the_relevant_passage_first() -> None:
    reranker = _reranker_or_skip()
    candidates = [
        ("c1", "The board approved a dividend of 1.25 dollars per share."),
        ("c2", "Operating margin compressed to 22.4 percent from 24.1 percent."),
        ("c3", "Revenue for the quarter was 412 million dollars."),
    ]
    run = reranker.rerank("What happened to operating margin?", candidates)
    assert run.used_reranker is True
    assert run.results[0].chunk_id == "c2"
    assert run.results[0].score > run.results[1].score > run.results[2].score


def test_empty_candidates_short_circuits_without_loading_the_model() -> None:
    reranker = CrossEncoderReranker()
    run = reranker.rerank("anything", [])
    assert run.results == []
    assert run.used_reranker is False
    assert reranker._model is None, "the model must not be loaded for an empty candidate list"


def test_unavailable_model_degrades_to_original_order_not_a_crash() -> None:
    reranker = CrossEncoderReranker(model_name="this-model-does-not-exist/definitely-not-a-real-repo")
    candidates = [("a", "first"), ("b", "second")]
    run = reranker.rerank("query", candidates)
    assert run.used_reranker is False
    assert run.reason is not None
    assert [r.chunk_id for r in run.results] == ["a", "b"], "original order must be preserved on fallback"
