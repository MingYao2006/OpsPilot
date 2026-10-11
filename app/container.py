"""应用容器：在 lifespan 中一次性装配所有单例，避免每次请求重复初始化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.services.embeddings import Embedder, build_embedder
from app.services.knowledge import KnowledgeBase
from app.services.llm import LLMClient, build_llm
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.fusion import fuse
from app.services.retrieval.rerank import rerank
from app.services.tickets import TicketStore
from app.services.vector_store.base import VectorStore
from app.services.vector_store.factory import build_vector_store
from app.utils.logging import get_logger
from app.services.agent import build_agent_graph

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    bm25: BM25Index
    knowledge: KnowledgeBase
    llm: LLMClient | None
    tickets: TicketStore
    notes: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    startup_summary: dict[str, Any] = field(default_factory=dict)
    agent_graph: Any = None

    @property
    def degraded(self) -> bool:
        return bool(self.notes)

    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def retrieve(self, question: str, *, top_k: int | None = None, service: str | None = None, use_rerank: bool = True):
        settings = self.settings
        top_k = top_k or settings.top_k
        candidate_k = max(top_k, settings.top_k)

        vector = self.embedder.embed_query(question)
        vec_hits = self.store.search(vector, top_k=candidate_k, service=service)
        vec_hits = [h for h in vec_hits if h.score >= settings.min_relevance]

        bm25_hits = self.bm25.search(question, top_k=candidate_k, service=service)

        fused = fuse([vec_hits, bm25_hits], k=settings.rrf_k, top_k=candidate_k)  # 粗排多召回

        if use_rerank:
            return rerank(question, fused, top_n=settings.rerank_top_n)[:top_k]  # 精排再截
        return fused[:top_k]

def build_container(settings: Settings) -> Container:
    embedder, embedder_notes = build_embedder(settings)
    store, store_notes = build_vector_store(settings, embedder.dim)
    bm25 = BM25Index(store)
    knowledge = KnowledgeBase(store, embedder, settings)
    llm, llm_notes = build_llm(settings)
    tickets = TicketStore(settings)
    if llm is None:
        log.info("LLM 离线（by design）：%s", "；".join(llm_notes))
    notes = list(embedder_notes) + list(store_notes)
    if notes:
        log.warning("检测到降级：%s", "；".join(notes))

    container= Container(
        settings=settings,
        embedder=embedder,
        store=store,
        bm25=bm25,
        knowledge=knowledge,
        tickets=tickets,
        llm=llm,
        notes=notes,
    )
    container.agent_graph = build_agent_graph(
        llm=llm, tickets=tickets,  search=lambda q, *, top_k=None, service=None: container.retrieve(
            q, top_k=top_k, service=knowledge.set_service_scope(service)
        ),settings=settings
    )

    return container
