"""SSE 流式接口测试。

难点在于：流式响应不能用普通接口的方式测。
- 普通接口：response.json() 拿完整对象
- 流式接口：TestClient 会把流读完再返回，但响应体是 SSE 文本，
  必须自己按 `data: <json>\\n\\n` 的协议切分并解析

LLM 通道在测试环境被钉死为 fallback（container.llm 为 None），
所以这里用两种方式覆盖：
1. 不替换 → 测「LLM 未就绪」的降级路径
2. 用 dependency_overrides 注入假流式客户端 → 测正常流式路径
"""
from __future__ import annotations

import copy
import json

import pytest

from app.api.deps import get_container


# ---------------- 工具 ----------------

def parse_sse(body: str) -> list[dict]:
    """把 SSE 响应体切成事件列表。

    协议：每个事件形如 `data: {...}\\n\\n`，事件之间用空行分隔。
    必须严格按这个格式解析，才能验证服务端格式没写错。
    """
    events: list[dict] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"事件必须以 'data: ' 开头，实际: {block[:40]!r}"
        events.append(json.loads(block[len("data: "):]))
    return events


def types_of(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


class FakeStreamingLLM:
    """最小可用的流式 LLM 替身：只实现 chat_stream。"""

    name = "fake-stream"

    def __init__(self, pieces: list[str] | None = None, *, raises: Exception | None = None):
        self.pieces = pieces if pieces is not None else ["支付", "网关", "连接池"]
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    def chat_stream(self, system: str, user: str):
        self.calls.append((system, user))
        if self.raises is not None:
            raise self.raises
        yield from self.pieces


# ---------------- fixture ----------------

@pytest.fixture
def stream_app(client, container):
    """返回 (client, 注入假 LLM 的函数)，用完自动清理 override。"""
    from app.main import app

    def _install(fake) -> None:
        fake_container = copy.copy(container)
        fake_container.llm = fake
        app.dependency_overrides[get_container] = lambda: fake_container

    yield client, _install
    app.dependency_overrides.pop(get_container, None)


# ---------------- 正常流式路径 ----------------

class TestStreamSuccess:
    def test_content_type_is_event_stream(self, stream_app):
        client, install = stream_app
        install(FakeStreamingLLM())
        resp = client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_anti_buffering_header_present(self, stream_app):
        """没有这个头，Nginx 会攒批转发，流式效果在部署后失效。"""
        client, install = stream_app
        install(FakeStreamingLLM())
        resp = client.post("/v1/ask/stream", json={"question": "x"})
        assert resp.headers.get("x-accel-buffering") == "no"
        assert "no-cache" in resp.headers.get("cache-control", "")

    def test_event_order(self, stream_app):
        """顺序必须是 meta → sources → token* → done。"""
        client, install = stream_app
        install(FakeStreamingLLM(["甲", "乙"]))
        resp = client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"})
        types = types_of(parse_sse(resp.text))

        assert types[0] == "meta"
        assert types[1] == "sources", "引用必须先于正文，让用户立刻看到来源"
        assert types[-1] == "done"
        assert types.count("token") == 2
        assert "error" not in types

    def test_sources_before_tokens(self, stream_app):
        """回归锁：sources 必须在第一个 token 之前。

        检索 ~0.6s、生成 3s+，先发引用能让用户提前看到「参考了哪几篇」。
        若顺序反了，用户要等整段答案出来才看到来源，体感差很多。
        """
        client, install = stream_app
        install(FakeStreamingLLM(["甲", "乙"]))
        types = types_of(parse_sse(
            client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"}).text
        ))
        assert types.index("sources") < types.index("token")

    def test_tokens_join_to_full_text(self, stream_app):
        client, install = stream_app
        install(FakeStreamingLLM(["支付", "网关", "连接池"]))
        events = parse_sse(
            client.post("/v1/ask/stream", json={"question": "x"}).text
        )
        text = "".join(e["text"] for e in events if e["type"] == "token")
        assert text == "支付网关连接池"

    def test_sources_carry_citations(self, stream_app):
        """检索必须真的跑过：sources 里应有引用条目且带 index/title。"""
        client, install = stream_app
        install(FakeStreamingLLM())
        events = parse_sse(
            client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"}).text
        )
        sources = next(e for e in events if e["type"] == "sources")
        assert sources["citations"], "应检索到资料"
        first = sources["citations"][0]
        assert first["index"] == 1
        assert first["title"]
        assert first["source"]

    def test_done_carries_latency(self, stream_app):
        client, install = stream_app
        install(FakeStreamingLLM())
        events = parse_sse(
            client.post("/v1/ask/stream", json={"question": "x"}).text
        )
        done = next(e for e in events if e["type"] == "done")
        assert isinstance(done["latency_ms"], int)
        assert done["latency_ms"] >= 0

    def test_prompt_contains_retrieved_context(self, stream_app):
        """LLM 必须拿到检索到的资料，否则它只能回答「资料中没有」。"""
        client, install = stream_app
        fake = FakeStreamingLLM()
        install(fake)
        client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"})

        assert fake.calls, "chat_stream 应被调用一次"
        system, user = fake.calls[0]
        assert "【资料】" in user
        assert "【问题】" in user
        assert "[1]" in user, "资料应带编号，供 LLM 标注引用"

    def test_chinese_not_escaped(self, stream_app):
        """ensure_ascii=False：中文必须原样输出，不能变成 \\u652f\\u4ed8。"""
        client, install = stream_app
        install(FakeStreamingLLM(["支付"]))
        resp = client.post("/v1/ask/stream", json={"question": "x"})
        assert "支付" in resp.text
        assert "\\u652f" not in resp.text


# ---------------- 降级与异常路径 ----------------

class TestStreamDegradation:
    def test_llm_none_yields_error_then_done(self, client):
        """测试环境 LLM 固定为 fallback，正好覆盖「未就绪」降级路径。"""
        resp = client.post("/v1/ask/stream", json={"question": "连接池耗尽怎么办"})
        assert resp.status_code == 200
        types = types_of(parse_sse(resp.text))
        assert types[0] == "meta"
        assert "error" in types
        assert types[-1] == "done", "出错也必须发 done，否则前端永远等下去"

    def test_llm_exception_yields_error_then_done(self, stream_app):
        client, install = stream_app
        install(FakeStreamingLLM(raises=RuntimeError("上游超时")))
        events = parse_sse(
            client.post("/v1/ask/stream", json={"question": "x"}).text
        )
        types = types_of(events)
        assert "error" in types
        assert types[-1] == "done"
        err = next(e for e in events if e["type"] == "error")
        assert "上游超时" in err["message"]

    def test_empty_question_rejected(self, client):
        resp = client.post("/v1/ask/stream", json={"question": ""})
        assert resp.status_code == 422
