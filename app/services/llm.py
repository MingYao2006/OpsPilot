"""可插拔 LLM 客户端（与 Embedding 通道同构：协议 + 降级 + notes）。

llm 不可用时 build_llm 返回 None，由上层（/v1/ask）降级为抽取式/拒答，绝不联网探测。
"""
from __future__ import annotations

import json
from typing import Protocol, Iterator, AsyncIterator

import httpx

from app.config import Settings
from app.utils.logging import get_logger

log = get_logger(__name__)

# OpenAI 兼容流式响应的一行形如：data: {"choices":[{"delta":{"content":"支"}}]}
# 终止标记是 data: [DONE]；另外流里还会有空行与心跳行，都要跳过。
_SSE_PREFIX = "data: "
_SSE_DONE = "data: [DONE]"


def _parse_sse_line(line: str) -> str | None:
    """解析一行 SSE，返回增量文本；非内容行返回 None。

    同步与异步两条流式路径共用，避免两份实现漂移。
    注意流式用 delta 而不是 message，且首帧只带 role 没有 content。
    """
    if not line or not line.startswith(_SSE_PREFIX):
        return None
    if line.strip() == _SSE_DONE:
        return None
    try:
        payload = json.loads(line[len(_SSE_PREFIX):])
        delta = payload["choices"][0].get("delta", {})
    except (json.JSONDecodeError, KeyError, IndexError):
        # 单行损坏不应中断整条流：跳过继续读下一行
        return None
    content = delta.get("content")
    return content or None


class LLMClient(Protocol):
    name: str

    def chat(self, system: str, user: str) -> str: ...

    def health(self) -> dict: ...

    def chat_with_tools(self, system: str, user: str, tools: list[dict]) -> dict: ...

    async def chat_stream_async(self, system: str, user: str) -> AsyncIterator[str]: ...

class OpenAICompatibleLLM:
    """DeepSeek / OpenAI / 任意 OpenAI 兼容端点共用一套（/chat/completions）。"""

    def __init__(self, api_key, model, base_url, temperature=0.1, timeout=90.0, json_mode=False, max_tokens=None):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self.name = f"openai-compatible:{model}"

    def _base_payload(self) -> dict:
        payload: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [],
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        return payload

    def chat(self, system: str, user: str, *, json_mode: bool | None = None) -> str:
        payload = self._base_payload()
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        use_json = self.json_mode if json_mode is None else json_mode
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("LLM 返回空内容")
        return content

    def health(self) -> dict:
        return {"provider": "openai-compatible", "model": self.model}

    def chat_with_tools(self, system: str, user: str, tools: list[dict]) -> dict:
        payload = self._base_payload()
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        """同步流式：给同步调用方用（如脚本、测试）。

        注意：在网络等待期间会阻塞当前线程。异步接口（/v1/ask/stream）
        必须用 chat_stream_async，否则会卡住整个事件循环。
        """
        payload = self._base_payload()
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload["stream"] = True

        with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                content = _parse_sse_line(line)
                if content:
                    yield content

    async def chat_stream_async(self, system: str, user: str) -> AsyncIterator[str]:
        """异步流式：在等网络时让出事件循环，不阻塞其他请求。

        与 chat_stream 的差异只有三处：async with / aiter_lines() / async for。
        解析逻辑抽到 _parse_sse_line 复用，避免两份实现漂移。
        """
        payload = self._base_payload()
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    content = _parse_sse_line(line)
                    if content:
                        yield content



def build_llm(settings: Settings) -> tuple[LLMClient | None, list[str]]:
    notes: list[str] = []
    if not settings.llm_ready:
        notes.append("LLM 未就绪（无 api_key 或 provider=fallback），答案层将降级为抽取式/拒答")
        return None, notes
    target = settings.resolve_llm()
    client = OpenAICompatibleLLM(
        api_key=target["api_key"],
        model=target["model"],
        base_url=target["base_url"],
        temperature=target["temperature"],
        timeout=target["timeout"],
        json_mode=target["json_mode"],
        max_tokens=target.get("max_tokens"),
    )
    log.info("LLM 生效：%s @ %s", target["provider"], target["base_url"])
    return client, notes

