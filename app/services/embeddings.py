"""可插拔的 Embedding 提供方（与 LLM 通道解耦）。

local   —— 内置特征哈希向量器，零依赖、可离线，用于演示与测试
ollama  —— 本地模型（nomic-embed-text 等）
openai  —— 任意 OpenAI 兼容端点（SiliconFlow / DashScope / Jina / 智谱 / vLLM 等）

注意：DeepSeek 目前不提供 embedding 接口，因此 Embedding 必须单独选择来源。
任一远端提供方不可用时会自动降级为 local，并在 /healthz 中如实上报。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from app.config import Settings
from app.utils.logging import get_logger
from app.utils.text import normalize, tokenize

log = get_logger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def health(self) -> dict: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0:
        return vector
    return [v / norm for v in vector]


class LocalHashingEmbedder:
    """特征哈希（hashing trick）+ 次线性 TF + L2 归一化。

    这不是语义模型，但具备两个关键性质：确定性、同词同维。
    配合 BM25 混合检索后在运维术语上有可用的召回率，且完全离线。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"local-hashing-{dim}"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dim
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        return _l2_normalize(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def health(self) -> dict:
        return {"provider": "local", "dim": self.dim, "offline": True}


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, dim: int = 768) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.name = f"ollama:{model}"

    def _post(self, inputs: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": inputs}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base_url}/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("ollama /api/embed 未返回 embeddings")
        self.dim = len(embeddings[0])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_l2_normalize(v) for v in self._post([normalize(t) for t in texts])]

    def embed_query(self, text: str) -> list[float]:
        return _l2_normalize(self._post([normalize(text)])[0])

    def health(self) -> dict:
        return {"provider": "ollama", "model": self.model, "dim": self.dim}


class OpenAICompatibleEmbedder:
    """任何遵循 OpenAI /v1/embeddings 协议的服务都可直接使用。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        dim: int = 512,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.dim = dim
        self.name = f"openai-compatible:{model}"

    def _post(self, inputs: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": inputs},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        items = sorted(data["data"], key=lambda d: d["index"])
        vectors = [item["embedding"] for item in items]
        if not vectors:
            raise RuntimeError("embedding 端点未返回向量")
        self.dim = len(vectors[0])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_l2_normalize(v) for v in self._post([normalize(t) for t in texts])]

    def embed_query(self, text: str) -> list[float]:
        return _l2_normalize(self._post([normalize(text)])[0])

    def health(self) -> dict:
        return {"provider": "openai-compatible", "model": self.model, "dim": self.dim}


def build_embedder(settings: Settings) -> tuple[Embedder, list[str]]:
    """按配置构建 embedder；远端不可用时降级到 local 并返回告警列表。"""
    notes: list[str] = []

    if settings.embedding_provider == "ollama":
        candidate = OllamaEmbedder(
            settings.ollama_base_url, settings.embedding_model, settings.embedding_dim
        )
        try:
            candidate.embed_query("健康检查")
            return candidate, notes
        except Exception as exc:  # noqa: BLE001 - 需要吞掉任何网络/协议错误
            notes.append(f"ollama embedding 不可用，已降级为 local：{exc}")
            log.warning("ollama embedding 不可用，降级为 local：%s", exc)

    elif settings.embedding_provider == "openai":
        if not settings.embedding_api_key:
            notes.append("未配置 OPSPILOT_EMBEDDING_API_KEY，已降级为 local")
        else:
            candidate = OpenAICompatibleEmbedder(
                settings.embedding_api_key,
                settings.embedding_model,
                settings.embedding_base_url,
                settings.embedding_dim,
            )
            try:
                candidate.embed_query("健康检查")
                return candidate, notes
            except Exception as exc:  # noqa: BLE001
                notes.append(f"远端 embedding 不可用，已降级为 local：{exc}")
                log.warning("远端 embedding 不可用，降级为 local：%s", exc)

    return LocalHashingEmbedder(settings.embedding_dim), notes
