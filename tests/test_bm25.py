"""BM25 惰性自愈回归：重启后空手索引，靠 search() 的 count 指纹自动重建，无需显式 build。"""

from __future__ import annotations

from app.models import Chunk
from app.services.retrieval.bm25 import BM25Index
from app.services.vector_store.memory import MemoryVectorStore


def _chunk(cid: str, text: str, service: str | None = None) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=f"doc-{cid}", text=text, source=f"{cid}.md", service=service)


class TestBM25SelfHeal:
    def test_first_query_rebuilds_after_restart(self):
        # 模拟重启：库里已有数据，但 BM25 是全新实例（_built=False，索引空手）
        store = MemoryVectorStore(dim=8)
        store.upsert(
            [_chunk("a", "支付网关 连接池 耗尽"), _chunk("b", "数据库 主从 延迟")],
            [[0.0] * 8, [0.0] * 8],
        )
        bm25 = BM25Index(store)          # 不手动 build
        hits = bm25.search("连接池 耗尽")  # 首个查询应惰性建索引并命中
        assert hits and hits[0].chunk.chunk_id == "a"

    def test_query_on_empty_store_returns_nothing(self):
        # 空库：_n==0 / avgdl==0，入口挡住除零，返回空而不是崩
        bm25 = BM25Index(MemoryVectorStore(dim=8))
        assert bm25.search("任意查询") == []

    def test_new_ingest_triggers_rebuild(self):
        store = MemoryVectorStore(dim=8)
        store.upsert([_chunk("a", "支付网关 连接池")], [[0.0] * 8])
        bm25 = BM25Index(store)
        assert bm25.search("支付")       # 先建索引，_n=1
        store.upsert([_chunk("c", "Redis 缓存 击穿")], [[0.0] * 8])  # count 变化
        hits = bm25.search("缓存 击穿")   # 下次查询应自动重建并命中新块
        assert hits and hits[0].chunk.chunk_id == "c"