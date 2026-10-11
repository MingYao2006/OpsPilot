"""集中式配置：全部通过 OPSPILOT_ 前缀的环境变量覆盖。

设计要点：Embedding 与 LLM 是两条独立通道。
DeepSeek 只提供对话模型、不提供 embedding 接口，因此二者必须能分别配置，
例如「DeepSeek 做 LLM + 本地 Ollama 做 Embedding」或「DeepSeek + 任意 OpenAI 兼容 embedding 服务」。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorBackend = Literal["memory", "qdrant", "pgvector"]
EmbeddingProvider = Literal["local", "ollama", "openai"]
LLMProvider = Literal["auto", "deepseek", "openai", "ollama", "fallback"]

# 各 LLM 提供方的默认端点与模型，可被显式的 base_url / model 覆盖
LLM_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPSPILOT_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "OpsPilot"
    version: str = "0.1.0"
    env: str = "dev"
    log_level: str = "INFO"

    # ---------- 数据位置 ----------
    knowledge_dir: Path = Path("data/kb")
    state_dir: Path = Path("work/state")
    ticket_store: Path = Path("work/state/tickets.json")
    autoload_knowledge: bool = True

    # ---------- 向量库 ----------
    vector_backend: VectorBackend = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "opspilot"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/opspilot"
    pg_table: str = "opspilot_chunks"

    # ---------- Embedding（与 LLM 完全独立）----------
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 512
    embedding_batch_size: int = 64
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ---------- LLM ----------
    llm_provider: LLMProvider = "deepseek"
    llm_model: str = ""
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.1
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 2
    llm_json_mode: bool = True
    llm_max_tokens: int = 2048

    # ---------- 检索 ----------
    chunk_size: int = 480
    chunk_overlap: int = 80
    top_k: int = 8
    rerank_top_n: int = 5
    rrf_k: int = 60
    min_relevance: float = 0.06
    max_agent_steps: int = 6
    max_rewrite_rounds: int = 2

    # ---------- 可信与安全 ----------
    require_citation: bool = True
    min_confidence: float = 0.35
    min_citation_support: float = 0.25

    # ---------- 可选鉴权（留空表示不校验，仅建议 dev 使用）----------
    jwt_secret: str | None = None

    # ---------- 校验 ----------

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_be_smaller(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 480)
        if v >= size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def _dim_positive(cls, v: int) -> int:
        if v < 32:
            raise ValueError("embedding_dim 至少为 32")
        return v

    @model_validator(mode="after")
    def _fill_llm_defaults(self) -> "Settings":
        preset = LLM_PRESETS.get(self.llm_provider, {})
        if not self.llm_model:
            self.llm_model = preset.get("model", "deepseek-chat")
        if not self.llm_base_url:
            self.llm_base_url = preset.get("base_url")
        return self

    # ---------- 派生属性 ----------

    def resolve_llm(self) -> dict:
        """解析当前生效的 LLM 端点参数。"""
        preset = LLM_PRESETS.get(self.llm_provider, {})
        return {
            "provider": self.llm_provider,
            "base_url": self.llm_base_url or preset.get("base_url"),
            "model": self.llm_model or preset.get("model", "deepseek-chat"),
            "api_key": self.llm_api_key,
            "temperature": self.llm_temperature,
            "timeout": self.llm_timeout_s,
            "json_mode": self.llm_json_mode,
        }

    @property
    def llm_ready(self) -> bool:
        """当前 LLM 配置是否具备发起真实调用的条件。"""
        target = self.resolve_llm()
        if target["provider"] in ("ollama",):
            return bool(target["base_url"])
        if target["provider"] == "fallback":
            return False
        return bool(target["api_key"] and target["base_url"])

    # 创建目录
    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ticket_store.parent.mkdir(parents=True, exist_ok=True)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.jwt_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
