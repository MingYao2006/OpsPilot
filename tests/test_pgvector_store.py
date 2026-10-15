"""pgvector 后端测试：用替身 engine 覆盖 SQL 构造、参数绑定与维度/表名校验。

不连真实 PostgreSQL（CI 无服务），但覆盖本项目会写错的地方：
表名注入防护、向量字面量格式、service 过滤是否下推到 SQL、
score 归一化、以及 hnsw 索引建失败时的降级。

psycopg / sqlalchemy 是可选依赖（[pg] extra），未安装时整体跳过。
真实连通性由 README 的手工验收步骤覆盖。
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy", reason="需要 sqlalchemy（pip install -e '.[pg]'）")

from app.models import Chunk  # noqa: E402
from app.services.vector_store.pgvector import PgVectorStore, _vec_literal  # noqa: E402


def _chunk(cid="c1", doc="d1", text="文本", service=None) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc, text=text, title="标题", source="kb/x.md",
        service=service, ordinal=1,
        updated_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


class FakeResult:
    def __init__(self, rows=None, scalar=None, rowcount=0):
        self._rows = rows or []
        self._scalar = scalar
        self.rowcount = rowcount

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar


class FakeConn:
    def __init__(self, responses=None):
        self.executed: list[tuple[str, object]] = []
        self._responses = list(responses or [])

    def execute(self, statement, params=None):
        sql = str(statement)
        self.executed.append((sql, params))
        return self._responses.pop(0) if self._responses else FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEngine:
    def __init__(self, responses=None, fail_on=None):
        self.conn = FakeConn(responses)
        self.fail_on = fail_on or []
        self.began = 0

    @contextmanager
    def begin(self):
        self.began += 1
        yield self.conn


def _store(engine, table="opspilot_chunks", dim=3) -> PgVectorStore:
    store = PgVectorStore.__new__(PgVectorStore)   # 跳过真实 create_engine
    store.table = table
    store.dim = dim
    store.engine = engine
    return store


class TestTableNameValidation:
    """表名会直接拼进 SQL（标识符无法参数化），必须白名单校验。"""

    def test_rejects_injection_attempt(self):
        with pytest.raises(ValueError):
            PgVectorStore("postgresql://x/y", table="chunks; DROP TABLE users--")

    def test_rejects_quotes(self):
        with pytest.raises(ValueError):
            PgVectorStore("postgresql://x/y", table='ch"unks')

    def test_rejects_leading_digit(self):
        with pytest.raises(ValueError):
            PgVectorStore("postgresql://x/y", table="1chunks")

    def test_accepts_valid_identifier(self):
        store = PgVectorStore.__new__(PgVectorStore)
        store.table = "opspilot_chunks_2"
        assert store.table == "opspilot_chunks_2"


class TestVecLiteral:
    def test_format(self):
        assert _vec_literal([0.5, -1.25]) == "[0.50000000,-1.25000000]"

    def test_rounds_to_eight_places(self):
        assert _vec_literal([0.123456789]) == "[0.12345679]"

    def test_empty(self):
        assert _vec_literal([]) == "[]"


class TestEnsureSchema:
    def test_creates_extension_table_and_indexes(self):
        engine = FakeEngine()
        _store(engine).ensure_schema()
        sql = " ".join(s for s, _ in engine.conn.executed).lower()
        assert "create extension if not exists vector" in sql
        assert "create table if not exists opspilot_chunks" in sql
        assert "hnsw" in sql, "应尝试建 HNSW 索引"
        assert "vector(3)" in sql, "维度必须写进 DDL"

    def test_hnsw_failure_falls_back_instead_of_raising(self):
        """老版本 pgvector 不支持 hnsw：应降级为顺序扫描而不是启动失败。"""
        engine = FakeEngine(fail_on=["hnsw"])
        original = engine.conn.execute

        def flaky(statement, params=None):
            if "hnsw" in str(statement).lower():
                raise RuntimeError("hnsw not supported")
            return original(statement, params)

        engine.conn.execute = flaky
        _store(engine).ensure_schema()   # 不应抛异常


class TestSearch:
    def _rows(self):
        return [
            {
                "chunk_id": "c1", "doc_id": "d1", "text": "文本", "title": "标题",
                "source": "kb/x.md", "service": "pay-gateway", "tags": ["a"],
                "ordinal": 1, "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "score": 0.87,
            }
        ]

    def test_returns_scored_chunks(self):
        engine = FakeEngine([FakeResult(self._rows())])
        hits = _store(engine).search([0.1, 0.2, 0.3], top_k=5)
        assert len(hits) == 1
        assert hits[0].retriever == "vector"
        assert hits[0].chunk.service == "pay-gateway"
        assert hits[0].score == pytest.approx(0.87)

    def test_score_is_clamped_to_unit_range(self):
        rows = self._rows()
        rows[0]["score"] = 1.7
        engine = FakeEngine([FakeResult(rows)])
        assert _store(engine).search([0.1, 0.2, 0.3], top_k=5)[0].score == 1.0

    def test_service_filter_is_pushed_into_sql(self):
        engine = FakeEngine([FakeResult(self._rows())])
        _store(engine).search([0.1, 0.2, 0.3], top_k=5, service="pay-gateway")
        sql, params = engine.conn.executed[-1]
        assert "service" in sql.lower(), "service 过滤应下推到 SQL 而不是取回后过滤"
        assert params["service"] == "pay-gateway"

    def test_no_service_passes_none(self):
        engine = FakeEngine([FakeResult(self._rows())])
        _store(engine).search([0.1, 0.2, 0.3], top_k=5)
        _, params = engine.conn.executed[-1]
        assert params["service"] is None

    def test_top_k_is_bound_as_parameter(self):
        engine = FakeEngine([FakeResult(self._rows())])
        _store(engine).search([0.1, 0.2, 0.3], top_k=7)
        _, params = engine.conn.executed[-1]
        assert params["k"] == 7

    def test_cosine_operator_is_used(self):
        engine = FakeEngine([FakeResult(self._rows())])
        _store(engine).search([0.1, 0.2, 0.3], top_k=5)
        sql, _ = engine.conn.executed[-1]
        assert "<=>" in sql, "余弦距离用 <=> 运算符"


class TestWriteAndAggregate:
    def test_upsert_binds_one_row_per_chunk(self):
        engine = FakeEngine([FakeResult(rowcount=2)])
        chunks = [_chunk(cid="c1"), _chunk(cid="c2")]
        assert _store(engine).upsert(chunks, [[0.1] * 3, [0.2] * 3]) == 2
        _, rows = engine.conn.executed[-1]
        assert len(rows) == 2
        assert rows[0]["embedding"].startswith("[")

    def test_upsert_empty_is_noop(self):
        engine = FakeEngine()
        assert _store(engine).upsert([], []) == 0
        assert engine.conn.executed == []

    def test_upsert_uses_on_conflict(self):
        engine = FakeEngine([FakeResult(rowcount=1)])
        _store(engine).upsert([_chunk()], [[0.1] * 3])
        sql, _ = engine.conn.executed[-1]
        assert "on conflict" in sql.lower(), "重复入库必须幂等"

    def test_count_reads_scalar(self):
        engine = FakeEngine([FakeResult(scalar=42)])
        assert _store(engine).count() == 42

    def test_delete_document_returns_rowcount(self):
        engine = FakeEngine([FakeResult(rowcount=3)])
        assert _store(engine).delete_document("d1") == 3

    def test_list_documents_maps_rows(self):
        rows = [{"doc_id": "d1", "title": "T", "source": "s", "service": None, "chunks": 2}]
        engine = FakeEngine([FakeResult(rows)])
        docs = _store(engine).list_documents()
        assert docs == [{"doc_id": "d1", "title": "T", "source": "s", "service": None, "chunks": 2}]

    def test_all_chunks_rebuilds_models(self):
        rows = [{
            "chunk_id": "c1", "doc_id": "d1", "text": "t", "title": "T", "source": "s",
            "service": None, "tags": ["a"], "ordinal": 1,
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }]
        engine = FakeEngine([FakeResult(rows)])
        chunks = _store(engine).all_chunks()
        assert chunks[0].chunk_id == "c1" and chunks[0].tags == ["a"]

    def test_health_reports_table(self):
        store = _store(FakeEngine())
        assert store.health()["table"] == "opspilot_chunks"
