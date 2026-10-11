"""Pydantic v2 数据契约：对外 API Schema + Agent 结构化输出。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Priority = Literal["P1", "P2", "P3", "P4"]
Intent = Literal["retrieve", "create_ticket", "chat"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Chunk(BaseModel):
    """检索的最小单元。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    source: str = "unknown"
    service: str | None = None
    tags: list[str] = Field(default_factory=list)
    ordinal: int = 0
    updated_at: datetime = Field(default_factory=_utcnow)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: Literal["vector", "bm25", "rrf", "rerank", "tool"] = "rrf"


class DocumentIn(BaseModel):
    doc_id: str | None = Field(default=None, max_length=128)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    source: str = Field(default="manual", max_length=300)
    service: str | None = Field(default=None, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=32)
    updated_at: datetime | None = None


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=200)
    replace: bool = True


class IngestResult(BaseModel):
    doc_id: str
    title: str
    chunks: int
    replaced: bool


class IngestResponse(BaseModel):
    ingested: list[IngestResult]
    total_chunks: int


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source: str
    service: str | None = None
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    service: str | None = None
    use_rerank: bool = True


class Citation(BaseModel):
    index: int = Field(ge=1, description="正文中的引用序号，形如 [1]")
    doc_id: str
    title: str
    source: str
    chunk_id: str
    snippet: str
    score: float = 0.0


class QueryResponse(BaseModel):
    question: str
    hits: list[ScoredChunk]
    citations: list[Citation]
    latency_ms: int


class AgentStep(BaseModel):
    node: str
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)


class AnswerPayload(BaseModel):
    """LLM 必须按此契约返回；解析失败即触发重试或拒答。"""

    answer: str = Field(min_length=1)
    used_citations: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_human: bool = False
    follow_up: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("follow_up", mode="before")
    @classmethod
    def _tolerant_follow_up(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    service: str | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[str] = Field(default_factory=list, max_length=20)


class AgentResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    needs_human: bool
    follow_up: list[str]
    refused: bool
    intent: Intent
    tools_used: list[str]
    steps: list[AgentStep]
    latency_ms: int


class TicketIn(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    summary: str = Field(min_length=3, max_length=4000)
    priority: Priority = "P3"
    service: str | None = Field(default=None, max_length=100)


class Ticket(BaseModel):
    ticket_id: str
    title: str
    summary: str
    priority: Priority
    service: str | None = None
    status: Literal["open", "closed"] = "open"
    created_at: datetime = Field(default_factory=_utcnow)


class TicketListResponse(BaseModel):
    """工单列表信封：带总数，与 /v1/query 的"元信息+数据"风格一致，也为将来分页留口。"""

    total: int
    items: list[Ticket]


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["llm", "rules"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    details: dict[str, Any] = Field(default_factory=dict)
