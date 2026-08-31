from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .domain import Chunk


class FaissVectorStore:
    """Persistent FAISS cosine store with a deterministic NumPy fallback."""

    name = "faiss"

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "chunks.index"
        self.metadata_path = self.index_dir / "chunks.json"
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._records: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if self.metadata_path.exists():
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            from .domain import Chunk

            self._records = [Chunk.model_validate(item) for item in payload]
        if self.index_path.exists():
            try:
                import faiss

                index = faiss.read_index(str(self.index_path))
                self._vectors = np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32) if index.ntotal else np.empty((0, index.d), dtype=np.float32)
            except Exception:
                self._vectors = np.empty((0, 0), dtype=np.float32)
        numpy_path = self.index_dir / "chunks.npy"
        if not len(self._vectors) and numpy_path.exists():
            self._vectors = np.asarray(np.load(numpy_path), dtype=np.float32)
        if self._records and len(self._records) != len(self._vectors):
            raise RuntimeError("FAISS metadata/vector count mismatch; rebuild the local index")

    def _persist(self) -> None:
        self.metadata_path.write_text(json.dumps([record.model_dump(mode="json") for record in self._records], ensure_ascii=False, indent=2), encoding="utf-8")
        if not len(self._vectors):
            return
        try:
            import faiss

            index = faiss.IndexFlatIP(self._vectors.shape[1])
            index.add(self._vectors.astype(np.float32))
            faiss.write_index(index, str(self.index_path))
        except Exception:
            np.save(self.index_dir / "chunks.npy", self._vectors)

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.ndim == 2 and self._vectors.size else 0

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        replacement = {chunk.chunk_id: (chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)}
        retained = [(chunk, self._vectors[index]) for index, chunk in enumerate(self._records) if chunk.chunk_id not in replacement]
        combined = retained + list(replacement.values())
        self._records = [item[0] for item in combined]
        self._vectors = np.vstack([item[1] for item in combined]).astype(np.float32) if combined else np.empty((0, vectors.shape[1]), dtype=np.float32)
        self._persist()

    def search(self, vector: np.ndarray, k: int = 10, source_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self._records:
            return []
        query = vector.astype(np.float32)
        scores = self._vectors @ query
        allowed = set(source_ids or [])
        order = np.argsort(-scores)
        results: list[tuple[Chunk, float]] = []
        for index in order:
            chunk = self._records[int(index)]
            if allowed and chunk.source_id not in allowed:
                continue
            results.append((chunk, float(scores[int(index)])))
            if len(results) >= k:
                break
        return results

    def all_records(self) -> list[Chunk]:
        return list(self._records)

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        for index, chunk in enumerate(self._records):
            if chunk.chunk_id == chunk_id and index < len(self._vectors):
                return np.asarray(self._vectors[index], dtype=np.float32)
        return None


def vector_store_capabilities() -> list[dict[str, Any]]:
    """Report installed/available stores without masking missing services."""

    stores = [
        ("faiss", True, "dense + local persistence"),
        ("chroma", False, "optional adapter not installed"),
        ("qdrant", False, "optional Docker adapter not started"),
        ("pgvector", False, "optional PostgreSQL adapter not started"),
        ("weaviate", False, "optional Docker adapter not started"),
        ("milvus", False, "optional Docker adapter not started"),
        ("pinecone", False, "optional cloud adapter; not required for local release"),
    ]
    return [{"name": name, "implemented": implemented, "description": description} for name, implemented, description in stores]
