"""向量库测试（内存后端）。"""

from __future__ import annotations

import pytest

from app.models import Chunk
from app.services.vector_store.memory import MemoryVectorStore


def _chunk(chunk_id: str, text: str, doc_id: str = "d1", service: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        title="标题",
        source="kb/x.md",
        service=service,
    )


@pytest.fixture
def store() -> MemoryVectorStore:
    return MemoryVectorStore(dim=3)


class TestUpsertAndSearch:
    def test_upsert_then_count(self, store):
        written = store.upsert(
            [_chunk("c1", "甲"), _chunk("c2", "乙")],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        assert written == 2
        assert store.count() == 2

    def test_search_orders_by_similarity(self, store):
        store.upsert(
            [_chunk("c1", "甲"), _chunk("c2", "乙")],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        hits = store.search([0.9, 0.1, 0.0], top_k=2)
        assert hits[0].chunk.chunk_id == "c1"
        assert hits[0].score > hits[1].score

    def test_search_score_is_normalized_to_unit_range(self, store):
        store.upsert([_chunk("c1", "甲")], [[1.0, 0.0, 0.0]])
        hits = store.search([-1.0, 0.0, 0.0])
        assert hits[0].score == 0.0, "负相关必须被裁剪为 0，否则会污染相关性门限"

    def test_top_k_limits_results(self, store):
        store.upsert(
            [_chunk(f"c{i}", f"文本{i}") for i in range(5)],
            [[1.0, 0.0, 0.0] for _ in range(5)],
        )
        assert len(store.search([1.0, 0.0, 0.0], top_k=3)) == 3

    def test_service_filter_applied_before_ranking(self, store):
        store.upsert(
            [_chunk("c1", "甲", service="pay"), _chunk("c2", "乙", service="order")],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        hits = store.search([1.0, 0.0, 0.0], service="order")
        assert [h.chunk.chunk_id for h in hits] == ["c2"]

    def test_upsert_is_idempotent_by_chunk_id(self, store):
        store.upsert([_chunk("c1", "旧")], [[1.0, 0.0, 0.0]])
        store.upsert([_chunk("c1", "新")], [[0.0, 1.0, 0.0]])
        assert store.count() == 1
        assert store.all_chunks()[0].text == "新"

    def test_mismatched_lengths_rejected(self, store):
        with pytest.raises(ValueError):
            store.upsert([_chunk("c1", "甲")], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_dimension_mismatch_rejected(self, store):
        store.upsert([_chunk("c1", "甲")], [[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            store.search([1.0, 0.0])


class TestDocumentManagement:
    def test_delete_document_removes_all_its_chunks(self, store):
        store.upsert(
            [_chunk("c1", "甲", "d1"), _chunk("c2", "乙", "d1"), _chunk("c3", "丙", "d2")],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        assert store.delete_document("d1") == 2
        assert store.count() == 1

    def test_delete_unknown_document_returns_zero(self, store):
        assert store.delete_document("missing") == 0

    def test_list_documents_aggregates_chunk_counts(self, store):
        store.upsert(
            [_chunk("c1", "甲", "d1"), _chunk("c2", "乙", "d1"), _chunk("c3", "丙", "d2")],
            [[1.0, 0.0, 0.0]] * 3,
        )
        grouped = {row["doc_id"]: row["chunks"] for row in store.list_documents()}
        assert grouped == {"d1": 2, "d2": 1}

    def test_reset_clears_everything(self, store):
        store.upsert([_chunk("c1", "甲")], [[1.0, 0.0, 0.0]])
        store.reset()
        assert store.count() == 0
        assert store.all_chunks() == []

    def test_health_reports_backend(self, store):
        info = store.health()
        assert info["backend"] == "memory"
        assert info["dim"] == 3
