"""RAG 问答接口：检索 + 生成（带引用与拒答判定）。

与 /v1/query 的区别：/v1/query 只检索、可评测；/v1/ask 会调用 LLM 生成结论。
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.deps import get_container, require_auth
from app.container import Container
from app.models import AgentRequest, AgentResponse, AgentStep
from app.services.answer import compose_answer
from app.services.citation import build_citations

router = APIRouter(prefix="/v1", tags=["ask"])


@router.post("/ask", response_model=AgentResponse, summary="RAG 问答（检索 + 生成）")
def ask(
    payload: AgentRequest,
    container: Container = Depends(get_container),
    _claims: dict = Depends(require_auth),
) -> AgentResponse:
    settings = container.settings
    started = time.perf_counter()

    top_k = payload.top_k or settings.top_k
    scope_service = container.knowledge.set_service_scope(payload.service)
    hits = container.retrieve(payload.question, top_k=top_k, service=scope_service)

    result = compose_answer(payload.question, hits, container.llm, settings)
    refused = container.llm is None or result.confidence < settings.min_confidence

    citations = build_citations(hits)
    steps = [
        AgentStep(node="retrieve", detail=f"命中 {len(hits)} 条"),
        AgentStep(node="answer", detail=f"refused={refused}, needs_human={result.needs_human}"),
    ]
    return AgentResponse(
        question=payload.question,
        answer=result.answer,
        citations=citations,
        confidence=result.confidence,
        needs_human=result.needs_human,
        follow_up=result.follow_up,
        refused=refused,
        intent="retrieve",
        tools_used=[],
        steps=steps,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )