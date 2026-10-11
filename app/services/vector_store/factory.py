"""向量库工厂：按配置选择后端，任一后端不可用则降级为 memory 并记录告警。"""

from __future__ import annotations

from app.config import Settings
from app.services.vector_store.base import VectorStore
from app.services.vector_store.memory import MemoryVectorStore
from app.utils.logging import get_logger

log = get_logger(__name__)


def build_vector_store(settings: Settings, dim: int) -> tuple[VectorStore, list[str]]:
    """返回 (store, notes)；notes 会出现在 /healthz 中，如实反映降级情况。"""
    notes: list[str] = []

    if settings.vector_backend == "qdrant":
        try:
            from app.services.vector_store.qdrant import QdrantVectorStore

            store = QdrantVectorStore(
                url=settings.qdrant_url,
                collection=settings.qdrant_collection,
                api_key=settings.qdrant_api_key,
            )
            store.ensure_collection(dim)
            log.info("向量库后端：qdrant (%s)", settings.qdrant_url)
            return store, notes
        except Exception as exc:  # noqa: BLE001 - 网络/依赖/维度不匹配都需降级
            notes.append(f"qdrant 不可用，已降级为 memory：{exc}")
            log.warning("qdrant 不可用，降级为 memory：%s", exc)

    elif settings.vector_backend == "pgvector":
        try:
            from app.services.vector_store.pgvector import PgVectorStore

            store = PgVectorStore(
                dsn=settings.database_url,
                table=settings.pg_table,
                dim=dim,
            )
            store.ensure_schema()
            log.info("向量库后端：pgvector (%s)", settings.pg_table)
            return store, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"pgvector 不可用，已降级为 memory：{exc}")
            log.warning("pgvector 不可用，降级为 memory：%s", exc)

    log.info("向量库后端：memory（维度 %d）", dim)
    return MemoryVectorStore(dim), notes
