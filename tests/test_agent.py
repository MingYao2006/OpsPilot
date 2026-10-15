"""/v1/agent 路由端到端测试：测试环境钉 fallback LLM，全离线、纯确定性。"""

from __future__ import annotations


class TestAgent:
    def test_create_ticket_offline(self, client):
        response = client.post("/v1/agent", json={"question": "支付网关挂了，帮我建工单"})
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "create_ticket"
        assert body["tools_used"] == ["create_ticket"]
        assert "已创建工单" in body["answer"]
        assert body["refused"] is False
        assert body["steps"], "应带回链路痕迹"

    def test_retrieve_offline_refuses_with_citations(self, client):
        response = client.post("/v1/agent", json={"question": "熔断错误率阈值是多少"})
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "retrieve"
        assert body["refused"] is True
        assert body["needs_human"] is True
        assert body["citations"], "离线拒答仍要如实带引用"

    def test_chat_no_tools(self, client):
        response = client.post("/v1/agent", json={"question": "今天天气怎么样"})
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "chat"
        assert body["tools_used"] == []
        assert body["refused"] is False

    def test_service_filter_applies_in_graph(self, client):
        """service 过滤的判据是「结果里不出现其它 service 的文档」，而不是写死文件名。

        原先写死 01/05/06 三个前缀，语料扩充后 17/28 也合法地挂了 pay-gateway，
        测试就假红了。改成从语料元数据推导期望集合，语料增删都不会误报，
        同时仍能抓住真正的越权（返回了别的 service 的文档）。
        """
        import pathlib

        from app.services.document_loader import read_text_file, service_from_meta

        kb_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "kb"
        pay_gateway = {
            p.name for p in kb_dir.glob("*.md")
            if service_from_meta(read_text_file(p)[0]) == "pay-gateway"
        }
        assert pay_gateway, "语料里应至少有一篇 pay-gateway 文档"

        response = client.post("/v1/agent", json={
            "question": "连接池耗尽怎么处理", "service": "pay-gateway"})
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "retrieve"
        assert body["citations"], "应命中 pay-gateway 手册"

        sources = [c["source"] for c in body["citations"]]
        leaked = [s for s in sources if s not in pay_gateway]
        assert not leaked, f"service 过滤未排除越权文档: {leaked}"

    def test_top_k_applies_in_graph(self, client):
        response = client.post("/v1/agent", json={
            "question": "熔断错误率阈值是多少", "top_k": 1})
        body = response.json()
        assert len(body["citations"]) == 1


class TestDraftTicket:
    """建单主轨镣铐：真 LLM 手气不可复现，替身演员上。"""

    def _make_graph(self, tmp_path, reply: str, tool_reply: dict | None = None):
        from app.config import Settings
        from app.services.agent import build_agent_graph
        from app.services.tickets import TicketStore

        class StubLLM:
            def chat(self, system: str, user: str) -> str:
                return reply

            def chat_with_tools(self, system: str, user: str, tools: list[dict]) -> dict:
                if tool_reply is None:
                    raise RuntimeError("本用例未提供 tool_reply，模拟 FC 不可用")
                return tool_reply

        settings = Settings(ticket_store=tmp_path / "t.json")
        return build_agent_graph(
            llm=StubLLM(),
            tickets=TicketStore(settings),
            search=lambda q: [],
            settings=settings,
        )

    def test_llm_extract_success_no_fallback_mark(self, tmp_path):
        graph = self._make_graph(
            tmp_path, '{"title": "数据库主从延迟飙升", "summary": "延迟超过30秒", "priority": "P1", "service": null}'
        )
        state = graph.invoke({"question": "数据库主从延迟高，建工单"})
        assert "兜底" not in state["answer"]
        assert state["ticket_id"]
        assert "数据库主从延迟飙升" in state["answer"]

    def test_llm_garbage_falls_back_with_mark(self, tmp_path):
        graph = self._make_graph(tmp_path, "抱歉，我无法处理这个请求")
        state = graph.invoke({"question": "数据库主从延迟高，建工单"})
        assert "（兜底建单）" in state["answer"]
        assert state["ticket_id"], "兜底也要真建出一张单"

    def test_real_function_call_primary_path(self, tmp_path):
        tool_reply = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {
                    "name": "create_ticket",
                    # 这个标题是 FC 专属，chat 那条路只会吐垃圾，故它出现即证明走了 FC
                    "arguments": '{"title": "FC专属标题-连接池告警", "summary": "线程池打满", "priority": "P2", "service": null}',
                },
            }],
        }
        graph = self._make_graph(tmp_path, reply="这不是 JSON，垃圾输出", tool_reply=tool_reply)
        state = graph.invoke({"question": "连接池打满了，建工单"})
        assert "FC专属标题-连接池告警" in state["answer"]  # 只有 FC 能给
        assert "兜底" not in state["answer"]
        assert state["ticket_id"]
