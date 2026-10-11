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
        response = client.post("/v1/agent", json={
            "question": "连接池耗尽怎么处理", "service": "pay-gateway"})
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "retrieve"
        assert body["citations"], "应命中 pay-gateway 手册"
        # pay-gateway 标签挂在 01/05/06 三篇手册上；不带 service 时 08-值班交接也会被召回，
        # 带上 service 后必须被排除——这才是过滤真正生效的证据
        sources = [c["source"] for c in body["citations"]]
        assert all(s.startswith(("01-", "05-", "06-")) for s in sources), sources
        assert not any("08-" in s for s in sources), f"service 过滤未排除越权文档: {sources}"

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
