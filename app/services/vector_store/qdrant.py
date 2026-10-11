"""Qdrant 后端：生产环境推荐，支持 HNSW 与 payload 过滤。"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import Chunk, ScoredChunk
from app.services.vector_store.base import VectorStore


def _to_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "title": chunk.title,
        "source": chunk.source,
        "service": chunk.service,
        "tags": list(chunk.tags),
        "ordinal": chunk.ordinal,
        "updated_at": chunk.updated_at.isoformat(),
    }


def _from_payload(payload: dict) -> Chunk:
    raw_ts = payload.get("updated_at")
    try:
        updated_at = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
    except ValueError:
        updated_at = datetime.now(timezone.utc)
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        title=payload.get("title", ""),
        source=payload.get("source", "unknown"),
        service=payload.get("service"),
        tags=list(payload.get("tags") or []),
        ordinal=int(payload.get("ordinal", 0)),
        updated_at=updated_at,
    )


class QdrantVectorStore(VectorStore):
    name = "qdrant"

    def __init__(self, url: str, collection: str, api_key: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30.0)

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client import models as qm

        self.dim = dim
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            info = self.client.get_collection(self.collection)
            params = info.config.params.vectors
            actual = getattr(params, "size", None)
            if actual and int(actual) != dim:
                raise ValueError(
                    f"Qdrant 集合 {self.collection} 维度为 {actual}，与当前 embedding 维度 {dim} 不一致；"
                    "请删除集合或更换 OPSPILOT_QDRANT_COLLECTION"
                )
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for field in ("doc_id", "service"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from qdrant_client import models as qm

        points = [
            qm.PointStruct(id=_point_id(chunk.chunk_id), vector=vector, payload=_to_payload(chunk))
            for chunk, vector in zip(chunks, vectors)
        ]
        if not points:
            return 0
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 8,
        service: str | None = None,
    ) -> list[ScoredChunk]:
        from qdrant_client import models as qm

        query_filter = None
        if service:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="service", match=qm.MatchValue(value=service))]
            )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            points = response.points
        else:  # pragma: no cover - 兼容旧版 qdrant-client
            points = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        hits: list[ScoredChunk] = []
        for point in points:
            hits.append(
                ScoredChunk(
                    chunk=_from_payload(point.payload or {}),
                    score=max(0.0, min(1.0, float(point.score))),
                    retriever="vector",
                )
            )
        return hits

    def delete_document(self, doc_id: str) -> int:
        from qdrant_client import models as qm

        before = self.count()
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )
        return max(0, before - self.count())

    def list_documents(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                entry = grouped.setdefault(
                    payload.get("doc_id", "?"),
                    {
                        "doc_id": payload.get("doc_id", "?"),
                        "title": payload.get("title", ""),
                        "source": payload.get("source", ""),
                        "service": payload.get("service"),
                        "chunks": 0,
                    },
                )
                entry["chunks"] += 1
            if offset is None:
                break
        return sorted(grouped.values(), key=lambda d: d["doc_id"])

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(_from_payload(r.payload or {}) for r in records)
            if offset is None:
                break
        return chunks

    def reset(self) -> None:
        self.client.delete_collection(self.collection)
        self.ensure_collection(self.dim)


def _point_id(chunk_id: str) -> int:
    """Qdrant 仅接受 uint64 或 UUID 作为点 ID，这里把 sha1 前缀映射为 uint64。"""
    return int(chunk_id[:16], 16) % (2**63)
