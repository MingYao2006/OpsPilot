"""检索接口。

/v1/query 刻意不调用 LLM —— 它是评测基线，用于单独衡量检索质量。
M2 已接入「向量 + BM25 + RRF」混合检索。
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from app.api.deps import get_container, require_auth
from app.container import Container
from app.models import QueryRequest, QueryResponse
from app.services.citation import build_citations

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse, summary="纯检索（不调用 LLM）")
def query(
    payload: QueryRequest,
    container: Container = Depends(get_container),
    _claims: dict = Depends(require_auth),
) -> QueryResponse:
    settings = container.settings
    started = time.perf_counter()

    top_k = payload.top_k or settings.top_k

    scope_service = container.knowledge.set_service_scope(payload.service)
    hits = container.retrieve(payload.question, top_k=top_k, service=scope_service, use_rerank=payload.use_rerank)

    return QueryResponse(
        question=payload.question,
        hits=hits,
        citations=build_citations(hits),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
