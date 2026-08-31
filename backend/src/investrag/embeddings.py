from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model: str
    fallback: bool


class EmbeddingProvider:
    """BGE-M3 when available, deterministic hashing fallback for offline tests."""

    def __init__(self, model_name: str, mode: str = "auto", fallback_dimension: int = 512) -> None:
        self.model_name = model_name
        self.mode = mode
        self.fallback_dimension = fallback_dimension
        self._model: Any = None
        self._fallback_reason: str | None = None

    @property
    def fallback(self) -> bool:
        return self.mode == "hash" or self._fallback_reason is not None

    @property
    def state(self) -> Literal["configured", "ready", "fallback"]:
        if self.fallback:
            return "fallback"
        if self._model is not None:
            return "ready"
        return "configured"

    @property
    def active_model_name(self) -> str:
        return f"hash-{self.fallback_dimension}" if self.fallback else self.model_name

    @property
    def dimension(self) -> int:
        if self._model is not None:
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension:
                return int(dimension)
        return self.fallback_dimension

    def _load(self) -> None:
        if self._model is not None or self._fallback_reason is not None:
            return
        if self.mode == "hash":
            self._fallback_reason = "hash mode requested"
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # local/offline machines should remain usable
            self._fallback_reason = f"embedding model unavailable: {exc.__class__.__name__}"

    def _hash_embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.fallback_dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower())
            for token in tokens:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.fallback_dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self._load()
        if self._model is None:
            return EmbeddingResult(
                self._hash_embed(texts), f"hash-{self.fallback_dimension}", True
            )
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return EmbeddingResult(np.asarray(vectors, dtype=np.float32), self.model_name, False)
