"""启发式重排：用查询-内容词覆盖率对 RRF 粗排结果做精排（零依赖、离线）。"""
from __future__ import annotations

from app.models import ScoredChunk
from app.utils.text import lexical_overlap


def rerank(query: str, candidates: list[ScoredChunk], *, top_n: int) -> list[ScoredChunk]:
    """RRF 粗排榜 → 按查询覆盖率精排，取前 top_n。retriever 标记 "rerank"。"""
    scored = []
    for hit in candidates:
        coverage = lexical_overlap(hit.chunk.text, query)   # 查询词被该块覆盖的比例
        scored.append((coverage, hit))
    scored.sort(key=lambda p: (p[0], p[1].score), reverse=True)   # 主键覆盖率, 次键RRF
    reranked=[]
    for coverage, hit in scored[:top_n]:
        reranked.append(ScoredChunk(chunk=hit.chunk, score=coverage, retriever="rerank"))
    #         chunk 用 hit.chunk，score 用 coverage，retriever="rerank"，收进列表返回
    return reranked