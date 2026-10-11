"""API 集成测试：走完整的 lifespan（含知识库自动加载），不 mock 内部实现。"""

from __future__ import annotations

import pytest

MARKDOWN_UPLOAD = """---
title: 单元测试用文档
service: unit-test-svc
tags: [测试]
---
这是一篇用于集成测试的文档。
它的关键词是「凤凰指标」与「9876」。
"""


class TestProbes:
    def test_healthz_ok_and_reports_actual_backends(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "ok"
        assert body["details"]["vector_backend"]["backend"] == "memory"
        assert body["details"]["embedder"]["provider"] == "local"
        assert body["details"]["llm"]["provider"] == "fallback"
        assert body["details"]["llm"]["ready"] is False

    def test_readyz_reports_loaded_knowledge(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "ok"
        assert body["details"]["chunks"] > 0
        assert body["details"]["problems"] == []

    def test_request_id_header_is_returned(self, client):
        response = client.get("/healthz", headers={"x-request-id": "trace-abc"})
        assert response.headers["x-request-id"] == "trace-abc"
        assert "x-process-time-ms" in response.headers


class TestQuery:
    def test_retrieves_expected_document(self, client):
        response = client.post("/v1/query", json={"question": "熔断错误率阈值是多少"})
        assert response.status_code == 200

        body = response.json()
        assert body["hits"], "应当检索到结果"
        assert body["hits"][0]["chunk"]["source"].startswith("05")

    def test_citations_align_with_hits(self, client):
        body = client.post("/v1/query", json={"question": "发布灰度比例是多少"}).json()
        assert len(body["citations"]) == len(body["hits"])
        assert [c["index"] for c in body["citations"]] == list(range(1, len(body["hits"]) + 1))
        for hit, citation in zip(body["hits"], body["citations"]):
            assert citation["chunk_id"] == hit["chunk"]["chunk_id"]

    def test_top_k_is_respected(self, client):
        body = client.post("/v1/query", json={"question": "数据库", "top_k": 2}).json()
        assert len(body["hits"]) <= 2

    def test_empty_question_rejected(self, client):
        assert client.post("/v1/query", json={"question": ""}).status_code == 422

    def test_missing_field_rejected(self, client):
        assert client.post("/v1/query", json={}).status_code == 422

    def test_service_filter_excludes_other_services(self, client):
        body = client.post(
            "/v1/query", json={"question": "连接池耗尽", "service": "不存在的服务"}
        ).json()
        assert body["hits"] == []

    def test_latency_is_reported(self, client):
        body = client.post("/v1/query", json={"question": "限流阈值"}).json()
        assert body["latency_ms"] >= 0


class TestAsk:
    def test_offline_refuses_but_keeps_citations(self, client):
        response = client.post("/v1/ask", json={"question": "熔断错误率阈值是多少"})
        assert response.status_code == 200

        body = response.json()
        # fallback 环境下 container.llm is None → compose_answer 走 _refuse
        assert body["refused"] is True
        assert body["needs_human"] is True
        assert body["answer"], "拒答也必须给一句说明"
        assert body["intent"] == "retrieve"
        assert body["tools_used"] == []
        assert body["steps"], "应带回链路痕迹"
        # 空1：断言离线作答仍带引用出处（citations 非空）
        assert body["citations"], "离线作答仍带引用出处（citations 非空）"

    def test_ask_empty_question_rejected(self, client):
        assert client.post("/v1/ask", json={"question": ""}).status_code == 422


class TestDocumentLifecycle:
    def test_ingest_list_delete_roundtrip(self, client):
        payload = {
            "documents": [
                {
                    "title": "集成测试文档",
                    "text": "本文件用于验证入库、列举与删除。关键词：蓝色河马。" * 5,
                    "source": "tests/lifecycle.md",
                    "service": "unit-test-svc",
                    "tags": ["测试"],
                }
            ],
            "replace": True,
        }

        ingest = client.post("/v1/documents", json=payload)
        assert ingest.status_code == 200
        result = ingest.json()["ingested"][0]
        assert result["chunks"] >= 1
        doc_id = result["doc_id"]

        listed = client.get("/v1/documents").json()
        assert any(row["doc_id"] == doc_id for row in listed)

        assert client.delete(f"/v1/documents/{doc_id}").status_code == 200

        listed_after = client.get("/v1/documents").json()
        assert not any(row["doc_id"] == doc_id for row in listed_after)

    def test_reingest_same_content_is_idempotent(self, client):
        payload = {
            "documents": [
                {"title": "幂等测试", "text": "内容固定不变。" * 10, "source": "tests/idem.md"}
            ],
            "replace": True,
        }
        first = client.post("/v1/documents", json=payload).json()["ingested"][0]
        second = client.post("/v1/documents", json=payload).json()["ingested"][0]

        assert first["doc_id"] == second["doc_id"]
        assert first["chunks"] == second["chunks"]

        client.delete(f"/v1/documents/{first['doc_id']}")

    def test_delete_unknown_document_returns_404(self, client):
        assert client.delete("/v1/documents/does-not-exist").status_code == 404

    def test_empty_document_list_rejected(self, client):
        assert client.post("/v1/documents", json={"documents": []}).status_code == 422


class TestUpload:
    def test_uploads_markdown_and_makes_it_searchable(self, client):
        response = client.post(
            "/v1/documents/upload",
            files={"file": ("unit.md", MARKDOWN_UPLOAD.encode("utf-8"), "text/markdown")},
        )
        assert response.status_code == 200
        doc_id = response.json()["doc_id"]
        assert response.json()["title"] == "单元测试用文档"

        try:
            hits = client.post("/v1/query", json={"question": "凤凰指标"}).json()["hits"]
            assert hits, "上传的内容应当可被检索"
        finally:
            client.delete(f"/v1/documents/{doc_id}")

    def test_unsupported_file_type_rejected(self, client):
        response = client.post(
            "/v1/documents/upload",
            files={"file": ("data.bin", b"\x00\x01\x02", "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_empty_file_rejected(self, client):
        response = client.post(
            "/v1/documents/upload",
            files={"file": ("empty.md", b"", "text/markdown")},
        )
        assert response.status_code == 400
