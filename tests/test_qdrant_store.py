"""Qdrant 后端测试：用替身 client 覆盖 payload 往返、过滤、分页与维度校验。

不连真实 Qdrant（CI 无服务），但覆盖的是**本项目的适配逻辑**：
payload 编解码、service 过滤构造、scroll 分页、维度不一致的报错，
这些才是会写错的地方。真实连通性由 README 的手工验收步骤覆盖。

qdrant-client 是可选依赖（pyproject 的 [qdrant] extra），未安装时整体跳过；
装了即生效，不需要真服务。
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("qdrant_client", reason="需要 qdrant-client（pip install -e '.[qdrant]'）")

from app.models import Chunk  # noqa: E402
from app.services.vector_store.qdrant import (  # noqa: E402
    QdrantVectorStore,
    _from_payload,
    _point_id,
    _to_payload,
)


def _chunk(cid="c1", doc="d1", text="文本", service=None, tags=("a",)) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc, text=text, title="标题", source="kb/x.md",
        service=service, tags=list(tags), ordinal=2,
        updated_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


class FakeClient:
    """只实现被测代码用到的那部分 QdrantClient 接口。"""

    def __init__(self, existing_collections=None, vector_size=None):
        self.collections = list(existing_collections or [])
        self.vector_size = vector_size
        self.created = []
        self.upserted = []
        self.indexes = []
        self.deleted = []
        self.points = []          # 供 scroll/query 返回
        self.last_filter = None

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.collections])

    def get_collection(self, name):
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=self.vector_size)))
        )

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config))
        self.collections.append(collection_name)

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexes.append(field_name)

    def upsert(self, collection_name, points, wait):
        self.upserted.extend(points)

    def query_points(self, collection_name, query, limit, query_filter, with_payload):
        self.last_filter = query_filter
        pts = [SimpleNamespace(payload=p, score=0.9) for p in self.points[:limit]]
        return SimpleNamespace(points=pts)

    def delete(self, collection_name, points_selector, wait):
        self.deleted.append(points_selector)

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        recs = [SimpleNamespace(payload=p) for p in self.points]
        return recs, None

    def count(self, collection_name, exact):
        return SimpleNamespace(count=len(self.points))

    def delete_collection(self, name):
        self.collections.remove(name)


def _store(client, collection="opspilot", dim=3) -> QdrantVectorStore:
    store = QdrantVectorStore.__new__(QdrantVectorStore)   # 跳过 __init__ 的真实连接
    store.collection = collection
    store.client = client
    store.dim = dim
    return store


class TestPayloadRoundTrip:
    def test_all_fields_survive(self):
        original = _chunk(service="pay-gateway")
        restored = _from_payload(_to_payload(original))
        assert restored.chunk_id == original.chunk_id
        assert restored.doc_id == original.doc_id
        assert restored.text == original.text
        assert restored.title == original.title
        assert restored.source == original.source
        assert restored.service == original.service
        assert restored.tags == original.tags
        assert restored.ordinal == original.ordinal
        assert restored.updated_at == original.updated_at

    def test_missing_optional_fields_get_defaults(self):
        c = _from_payload({"chunk_id": "c", "doc_id": "d", "text": "t"})
        assert c.title == ""
        assert c.source == "unknown"
        assert c.service is None
        assert c.tags == []
        assert c.ordinal == 0

    def test_bad_timestamp_falls_back_to_now(self):
        c = _from_payload({"chunk_id": "c", "doc_id": "d", "text": "t", "updated_at": "not-a-date"})
        assert c.updated_at.tzinfo is not None

    def test_tags_are_copied_not_shared(self):
        """payload 里的 list 若被直接引用，改一处会污染另一处。"""
        payload = _to_payload(_chunk(tags=("x", "y")))
        c = _from_payload(payload)
        c.tags.append("z")
        assert payload["tags"] == ["x", "y"]


class TestPointId:
    def test_uint64_range(self):
        pid = _point_id("a" * 16)
        assert 0 <= pid < 2**63

    def test_deterministic(self):
        assert _point_id("0123456789abcdef") == _point_id("0123456789abcdef")

    def test_different_ids_differ(self):
        assert _point_id("0" * 16) != _point_id("f" * 16)


class TestEnsureCollection:
    def test_creates_when_absent_with_indexes(self):
        client = FakeClient()
        _store(client).ensure_collection(dim=8)
        assert client.created and client.created[0][0] == "opspilot"
        assert set(client.indexes) == {"doc_id", "service"}

    def test_idempotent_when_dimension_matches(self):
        client = FakeClient(existing_collections=["opspilot"], vector_size=8)
        _store(client).ensure_collection(dim=8)
        assert client.created == [], "已存在且维度一致时不应重复创建"

    def test_dimension_mismatch_raises_with_actionable_message(self):
        client = FakeClient(existing_collections=["opspilot"], vector_size=512)
        with pytest.raises(ValueError) as ei:
            _store(client).ensure_collection(dim=1024)
        msg = str(ei.value)
        assert "512" in msg and "1024" in msg
        assert "QDRANT_COLLECTION" in msg, "报错要给出可执行的处置建议"


class TestSearch:
    def test_service_filter_is_built(self):
        client = FakeClient()
        client.points = [_to_payload(_chunk())]
        _store(client).search([0.1, 0.2, 0.3], top_k=5, service="pay-gateway")
        assert client.last_filter is not None, "指定 service 时必须下推过滤条件"

    def test_no_filter_when_service_absent(self):
        client = FakeClient()
        client.points = [_to_payload(_chunk())]
        _store(client).search([0.1, 0.2, 0.3], top_k=5)
        assert client.last_filter is None

    def test_score_is_clamped_and_labelled(self):
        client = FakeClient()
        client.points = [_to_payload(_chunk())]
        hits = _store(client).search([0.1, 0.2, 0.3], top_k=5)
        assert hits and hits[0].retriever == "vector"
        assert 0.0 <= hits[0].score <= 1.0

    def test_respects_top_k(self):
        client = FakeClient()
        client.points = [_to_payload(_chunk(cid=f"c{i}")) for i in range(5)]
        assert len(_store(client).search([0.1, 0.2, 0.3], top_k=2)) == 2


class TestAggregates:
    def test_list_documents_counts_chunks_per_doc(self):
        client = FakeClient()
        client.points = [
            _to_payload(_chunk(cid="c1", doc="d1")),
            _to_payload(_chunk(cid="c2", doc="d1")),
            _to_payload(_chunk(cid="c3", doc="d2")),
        ]
        docs = _store(client).list_documents()
        assert [d["doc_id"] for d in docs] == ["d1", "d2"]
        assert docs[0]["chunks"] == 2
        assert docs[1]["chunks"] == 1

    def test_count_and_all_chunks(self):
        client = FakeClient()
        client.points = [_to_payload(_chunk(cid=f"c{i}")) for i in range(4)]
        store = _store(client)
        assert store.count() == 4
        assert len(store.all_chunks()) == 4

    def test_upsert_returns_written_count(self):
        client = FakeClient()
        chunks = [_chunk(cid="c1"), _chunk(cid="c2")]
        assert _store(client).upsert(chunks, [[0.1] * 3, [0.2] * 3]) == 2
        assert len(client.upserted) == 2

    def test_upsert_empty_is_noop(self):
        client = FakeClient()
        assert _store(client).upsert([], []) == 0
        assert client.upserted == []
