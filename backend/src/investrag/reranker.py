from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankedCandidate:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class RerankRun:
    results: list[RerankedCandidate]
    used_reranker: bool
    reason: str | None = None


class CrossEncoderReranker:
    """`mixedbread-ai/mxbai-rerank-base-v2` behind design §11 step 6's
    contextual-compression/reranking stage.

    Chosen over the design's original `BAAI/bge-reranker-v2-m3` (absent from current
    reranker trending) and over `nvidia/llama-nemotron-rerank-1b-v2` /
    `jinaai/jina-reranker-v2-base-multilingual` (both require `trust_remote_code=True`
    - executing a vendor's custom modeling code - and Jina's is non-commercial
    licensed). mxbai-rerank-base-v2 is Apache-2.0, loads through the standard
    `sentence_transformers.CrossEncoder` API with no remote code execution, and is
    already exercised on this machine: verified 2026-08-30 on CPU, 0.44s to score 3
    candidates, correctly ranking a margin-specific passage above unrelated ones for a
    margin query.

    Defaults to CPU per design §19's VRAM budget - reranking runs alongside a resident
    generation model and embedder, and this model does not need a GPU to be fast
    enough for interactive use.
    """

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-base-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._unavailable_reason: str | None = None

    def _load(self) -> Any:
        if self._model is not None or self._unavailable_reason is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:  # offline test environments must still function
            self._unavailable_reason = f"{exc.__class__.__name__}: {exc}"
        return self._model

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> RerankRun:
        """`candidates` is a list of (chunk_id, text). Returns candidates sorted by
        relevance to `query`, or the original order (with `used_reranker=False` and a
        reason) if the model cannot be loaded - degradation is reported, never silent."""

        if not candidates:
            return RerankRun(results=[], used_reranker=False, reason="no candidates to rerank")
        model = self._load()
        if model is None:
            fallback = [RerankedCandidate(chunk_id=chunk_id, score=0.0) for chunk_id, _ in candidates]
            return RerankRun(results=fallback, used_reranker=False, reason=self._unavailable_reason)
        pairs = [(query, text) for _, text in candidates]
        scores = model.predict(pairs)
        scored = [
            RerankedCandidate(chunk_id=chunk_id, score=float(score))
            for (chunk_id, _), score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return RerankRun(results=scored, used_reranker=True)
