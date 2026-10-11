"""PostgreSQL + pgvector 后端：与业务库同源，运维成本最低。"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.models import Chunk, ScoredChunk
from app.services.vector_store.base import VectorStore

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _vec_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


class PgVectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, dsn: str, table: str = "opspilot_chunks", dim: int = 1536) -> None:
        from sqlalchemy import create_engine

        if not _IDENT.match(table):
            raise ValueError(f"非法表名：{table!r}（仅允许字母、数字与下划线）")

        self.table = table
        self.dim = dim
        self.engine = create_engine(dsn, pool_pre_ping=True, future=True)

    # ---------- schema ----------

    def ensure_schema(self) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        chunk_id   TEXT PRIMARY KEY,
                        doc_id     TEXT NOT NULL,
                        text       TEXT NOT NULL,
                        title      TEXT DEFAULT '',
                        source     TEXT DEFAULT '',
                        service    TEXT,
                        tags       JSONB DEFAULT '[]'::jsonb,
                        ordinal    INTEGER DEFAULT 0,
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        embedding  vector({self.dim}) NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {self.table}_doc_idx ON {self.table} (doc_id)")
            )
            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {self.table}_svc_idx ON {self.table} (service)")
            )
            try:
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {self.table}_vec_idx "
                        f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
                    )
                )
            except Exception:  # noqa: BLE001 - 老版本 pgvector 不支持 hnsw，退回顺序扫描
                pass

    # ---------- 写入 ----------

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from sqlalchemy import text

        if not chunks:
            return 0
        statement = text(
            f"""
            INSERT INTO {self.table}
                (chunk_id, doc_id, text, title, source, service, tags, ordinal, updated_at, embedding)
            VALUES
                (:chunk_id, :doc_id, :text, :title, :source, :service,
                 CAST(:tags AS jsonb), :ordinal, :updated_at, CAST(:embedding AS vector))
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                title = EXCLUDED.title,
                source = EXCLUDED.source,
                service = EXCLUDED.service,
                tags = EXCLUDED.tags,
                ordinal = EXCLUDED.ordinal,
                updated_at = EXCLUDED.updated_at,
                embedding = EXCLUDED.embedding
            """
        )
        import json

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "title": chunk.title,
                "source": chunk.source,
                "service": chunk.service,
                "tags": json.dumps(list(chunk.tags), ensure_ascii=False),
                "ordinal": chunk.ordinal,
                "updated_at": chunk.updated_at,
                "embedding": _vec_literal(vector),
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        with self.engine.begin() as conn:
            conn.execute(statement, rows)
        return len(rows)

    # ---------- 检索 ----------

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 8,
        service: str | None = None,
    ) -> list[ScoredChunk]:
        from sqlalchemy import text

        statement = text(
            f"""
            SELECT chunk_id, doc_id, text, title, source, service, tags, ordinal, updated_at,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM {self.table}
            WHERE (CAST(:service AS text) IS NULL OR service = CAST(:service AS text))
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
            """
        )
        with self.engine.begin() as conn:
            rows = (
                conn.execute(statement, {"vec": _vec_literal(vector), "service": service, "k": top_k})
                .mappings()
                .all()
            )

        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    title=row["title"] or "",
                    source=row["source"] or "unknown",
                    service=row["service"],
                    tags=list(row["tags"] or []),
                    ordinal=int(row["ordinal"] or 0),
                    updated_at=row["updated_at"] or datetime.now(timezone.utc),
                ),
                score=max(0.0, min(1.0, float(row["score"] or 0.0))),
                retriever="vector",
            )
            for row in rows
        ]

    # ---------- 文档管理 ----------

    def delete_document(self, doc_id: str) -> int:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            result = conn.execute(
                text(f"DELETE FROM {self.table} WHERE doc_id = :doc_id"), {"doc_id": doc_id}
            )
        return int(result.rowcount or 0)

    def list_documents(self) -> list[dict]:
        from sqlalchemy import text

        statement = text(
            f"""
            SELECT doc_id,
                   MIN(title)  AS title,
                   MIN(source) AS source,
                   MIN(service) AS service,
                   COUNT(*)    AS chunks
            FROM {self.table}
            GROUP BY doc_id
            ORDER BY doc_id
            """
        )
        with self.engine.begin() as conn:
            rows = conn.execute(statement).mappings().all()
        return [
            {
                "doc_id": row["doc_id"],
                "title": row["title"] or "",
                "source": row["source"] or "",
                "service": row["service"],
                "chunks": int(row["chunks"]),
            }
            for row in rows
        ]

    def count(self) -> int:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {self.table}")).scalar() or 0)

    def all_chunks(self) -> list[Chunk]:
        from sqlalchemy import text

        statement = text(
            f"SELECT chunk_id, doc_id, text, title, source, service, tags, ordinal, updated_at "
            f"FROM {self.table}"
        )
        with self.engine.begin() as conn:
            rows = conn.execute(statement).mappings().all()
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                title=row["title"] or "",
                source=row["source"] or "unknown",
                service=row["service"],
                tags=list(row["tags"] or []),
                ordinal=int(row["ordinal"] or 0),
                updated_at=row["updated_at"] or datetime.now(timezone.utc),
            )
            for row in rows
        ]

    def reset(self) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.table}"))
        self.ensure_schema()

    def health(self) -> dict:
        info = super().health()
        info["table"] = self.table
        return info
