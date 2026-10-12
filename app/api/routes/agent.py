"""Agent 接口：LangGraph 三岔路（知识问答 / 建工单 / 闲聊）。"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.deps import get_container, require_auth
from app.container import Container
from app.models import AgentRequest, AgentResponse, AgentStep

router = APIRouter(prefix="/v1", tags=["agent"])


@router.post("/agent", response_model=AgentResponse, summary="Agent：意图路由 + 知识问答/建工单")
def agent(
    payload: AgentRequest,
    container: Container = Depends(get_container),
    _claims: dict = Depends(require_auth),
) -> AgentResponse:
    settings = container.settings
    started = time.perf_counter()

    state = container.agent_graph.invoke({
        "question": payload.question,
        "service": payload.service,
        "top_k": payload.top_k,
        "history": payload.history,
    })
    intent = state["intent"]
    answer_confidence = state.get("answer_confidence", 0.0)
    refused = intent == "retrieve" and (container.llm is None or answer_confidence < settings.min_confidence)
    steps = [
        AgentStep(node="classify", detail=f"intent={intent}, source={state.get('source')}"),
        AgentStep(node=intent, detail=f"tools={state.get('tools_used', [])}"),
    ]
    return AgentResponse(
        question=payload.question,
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        confidence=answer_confidence,
        needs_human=state.get("needs_human", False),
        follow_up=state.get("follow_up", []),
        refused=refused,
        intent=intent,
        tools_used=state.get("tools_used", []),
        steps=steps,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )