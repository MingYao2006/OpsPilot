"""内存向量库：零依赖，用于本地开发、演示与测试。

采用暴力余弦检索。万级块以内延迟可忽略；更大规模请切换 qdrant / pgvector。
"""

from __future__ import annotations

import math
import threading

from app.models import Chunk, ScoredChunk
from app.services.vector_store.base import VectorStore


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致：{len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


class MemoryVectorStore(VectorStore):
    name = "memory"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        with self._lock:
            for chunk, vector in zip(chunks, vectors):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = list(vector)
        return len(chunks)

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 8,
        service: str | None = None,
    ) -> list[ScoredChunk]:
        with self._lock:
            items = list(self._vectors.items())
            chunks = dict(self._chunks)

        scored: list[ScoredChunk] = []
        for chunk_id, stored in items:
            chunk = chunks[chunk_id]
            if service and chunk.service != service:
                continue
            score = max(0.0, _cosine(vector, stored))
            scored.append(ScoredChunk(chunk=chunk, score=score, retriever="vector"))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            victims = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
            for cid in victims:
                self._chunks.pop(cid, None)
                self._vectors.pop(cid, None)
        return len(victims)

    def list_documents(self) -> list[dict]:
        with self._lock:
            chunks = list(self._chunks.values())

        grouped: dict[str, dict] = {}
        for chunk in chunks:
            entry = grouped.setdefault(
                chunk.doc_id,
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "service": chunk.service,
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        return sorted(grouped.values(), key=lambda d: d["doc_id"])

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def reset(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._vectors.clear()

    def health(self) -> dict:
        info = super().health()
        info["duplicate_chunk_ids"] = 0
        return info
