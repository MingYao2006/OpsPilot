"""SSE 流式接口测试。

流式响应不能用普通接口的方式测：响应体是 SSE 文本，
必须按 `data: <json>\\n\\n` 协议自己切分解析。

测试环境把 LLM 钉死为 fallback（container.llm 为 None），
所以分两类覆盖：
1. 不注入 → 测降级路径（meta → error → done）
2. 用 dependency_overrides 注入假流式客户端 → 测正常路径（含 sources）
"""
from __future__ import annotations

import copy
import json

import pytest

from app.api.deps import get_container


# ---------------- 工具 ----------------

def parse_sse(body: str) -> list[dict]:
    """把 SSE 响应体解析成事件列表。"""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"事件必须以 'data: ' 开头，实际: {block[:40]!r}"
        events.append(json.loads(block[len("data: "):]))
    return events


def types_of(body: str) -> list[str]:
    return [e["type"] for e in parse_sse(body)]


class FakeStreamingLLM:
    """假流式 LLM：chat_stream 返回可迭代字符串即可。

    记录 calls 以便断言「提示词里确实带了检索到的资料」。
    """

    name = "fake-stream"

    def __init__(self, pieces=None, *, raises=None):
        self.pieces = ["支付", "网关", "连接池"] if pieces is None else pieces
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    def chat_stream(self, system: str, user: str):
        self.calls.append((system, user))
        if self.raises is not None:
            raise self.raises
        yield from self.pieces


# ---------------- fixture ----------------

@pytest.fixture
def fake_llm_app(client, container):
    """返回 (client, install)，install(假LLM) 后接口就走正常流式路径。"""
    from app.main import app

    def _install(fake) -> None:
        fake_container = copy.copy(container)   # 浅拷贝，不动 session 级真容器
        fake_container.llm = fake
        app.dependency_overrides[get_container] = lambda: fake_container

    yield client, _install
    app.dependency_overrides.pop(get_container, None)   # 必须清理，否则污染其他测试


# ---------------- 响应头 ----------------

class TestHeaders:
    def test_content_type_is_event_stream(self, client):
        """SSE 必须返回 text/event-stream，浏览器才知道这是流。"""
        resp = client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_anti_buffering_header(self, client):
        """没有 X-Accel-Buffering，Nginx 会攒批转发，部署后流式直接失效。"""
        resp = client.post("/v1/ask/stream", json={"question": "x"})
        assert resp.headers.get("x-accel-buffering") == "no", (
            "缺少 X-Accel-Buffering: no —— 部署到 Nginx 后数据会被攒批转发，"
            "后端明明在流式发，用户还是等全部结束才看到"
        )
        assert "no-cache" in resp.headers.get("cache-control", "")


# ---------------- 降级路径（LLM 为 None）----------------

class TestDegradation:
    def test_event_order(self, client):
        """测试环境 LLM 固定 fallback，走降级路径。"""
        types = types_of(
            client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"}).text
        )
        assert types == ["meta", "error", "done"]

    def test_done_always_last(self, client):
        """出错也必须发 done，否则前端永远等下去。"""
        types = types_of(client.post("/v1/ask/stream", json={"question": "x"}).text)
        assert types[-1] == "done"


# ---------------- 正常路径（注入假 LLM）----------------

class TestSuccess:
    def test_sources_event_exists(self, fake_llm_app):
        """正常路径必须有 sources 事件，否则前端拿不到引用卡片。"""
        client, install = fake_llm_app
        install(FakeStreamingLLM())

        types = types_of(client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"}).text)
        assert "sources" in types, "缺少 sources 事件，前端拿不到引用卡片"

    def test_full_event_order(self, fake_llm_app):
        """顺序：meta → sources → token* → done。"""
        client, install = fake_llm_app
        install(FakeStreamingLLM(["甲", "乙"]))

        types = types_of(client.post("/v1/ask/stream", json={"question": "x"}).text)
        assert types[0] == "meta"
        assert types[1] == "sources"
        assert types[-1] == "done"
        assert types.count("token") == 2
        assert "error" not in types

    def test_sources_before_tokens(self, fake_llm_app):
        """回归锁：sources 必须先于 token。

        检索 ~0.6s、生成 3s+，先发引用能让用户立刻看到「参考了哪几篇」。
        顺序反了用户要等整段答案出来才看到来源，体感差很多。
        """
        client, install = fake_llm_app
        install(FakeStreamingLLM(["甲"]))

        types = types_of(client.post("/v1/ask/stream", json={"question": "x"}).text)
        assert types.index("sources") < types.index("token")

    def test_tokens_join_to_full_text(self, fake_llm_app):
        client, install = fake_llm_app
        install(FakeStreamingLLM(["支付", "网关", "连接池"]))

        events = parse_sse(client.post("/v1/ask/stream", json={"question": "x"}).text)
        text = "".join(e["text"] for e in events if e["type"] == "token")
        assert text == "支付网关连接池"

    def test_sources_carry_citations(self, fake_llm_app):
        """检索必须真跑过：sources 里应有引用条目。"""
        client, install = fake_llm_app
        install(FakeStreamingLLM())

        events = parse_sse(client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"}).text)
        sources = next(e for e in events if e["type"] == "sources")
        assert sources["citations"], "应检索到资料"
        first = sources["citations"][0]
        assert first["index"] == 1
        assert first["title"]
        assert first["source"]

    def test_done_carries_latency(self, fake_llm_app):
        client, install = fake_llm_app
        install(FakeStreamingLLM())

        events = parse_sse(client.post("/v1/ask/stream", json={"question": "x"}).text)
        done = next(e for e in events if e["type"] == "done")
        assert isinstance(done["latency_ms"], int)
        assert done["latency_ms"] >= 0

    def test_prompt_contains_retrieved_context(self, fake_llm_app):
        """LLM 必须拿到检索到的资料，否则只能回答「资料中没有」。"""
        client, install = fake_llm_app
        fake = FakeStreamingLLM()
        install(fake)

        client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"})
        assert fake.calls, "chat_stream 应被调用一次"
        _system, user = fake.calls[0]
        assert "【资料】" in user
        assert "【问题】" in user
        assert "[1]" in user, "资料应带编号，供 LLM 标注引用"

    def test_chinese_not_escaped(self, fake_llm_app):
        """ensure_ascii=False：中文原样输出，不能变成 \\u652f\\u4ed8。"""
        client, install = fake_llm_app
        install(FakeStreamingLLM(["支付"]))

        resp = client.post("/v1/ask/stream", json={"question": "x"})
        assert "支付" in resp.text
        assert "\\u652f" not in resp.text

    def test_llm_exception_yields_error_then_done(self, fake_llm_app):
        """LLM 流到一半抛错时，必须 error 后仍发 done。"""
        client, install = fake_llm_app
        install(FakeStreamingLLM(raises=RuntimeError("上游超时")))

        events = parse_sse(client.post("/v1/ask/stream", json={"question": "x"}).text)
        types = [e["type"] for e in events]
        assert "error" in types
        assert types[-1] == "done"
        err = next(e for e in events if e["type"] == "error")
        assert "上游超时" in err["message"]


# ---------------- 参数校验 ----------------

class TestValidation:
    def test_empty_question_rejected(self, client):
        resp = client.post("/v1/ask/stream", json={"question": ""})
        assert resp.status_code == 422
