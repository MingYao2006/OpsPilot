"""可插拔 LLM 客户端（与 Embedding 通道同构：协议 + 降级 + notes）。

llm 不可用时 build_llm 返回 None，由上层（/v1/ask）降级为抽取式/拒答，绝不联网探测。
"""
from __future__ import annotations

import json
from typing import Protocol, Iterator

import httpx

from app.config import Settings
from app.utils.logging import get_logger

log = get_logger(__name__)


class LLMClient(Protocol):
    name: str

    def chat(self, system: str, user: str) -> str: ...

    def health(self) -> dict: ...

    def chat_with_tools(self, system: str, user: str, tools: list[dict]) -> dict: ...


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
                if line=="":
                    continue
                if not line.startswith("data: "):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content")
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

