"""RRF 倒数排名融合：把多路检索结果翻译成共识排名。"""

from __future__ import annotations

from typing import List

from app.models import ScoredChunk


def fuse(
        rankings: list[list[ScoredChunk]], *, k: int = 60, top_k: int = 8, weights: list[float] | None = None
) -> list[ScoredChunk]:
    """多路榜单 → 一路融合榜。retriever 标记为 "rrf"。"""
    accum = {}
    if weights is None:
        weights = [1.0] * len(rankings)

    for i,ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            rrf = weights[i] / (k + rank)
            if hit.chunk.chunk_id not in accum:
               accum[hit.chunk.chunk_id] = [rrf, hit]
            else:
                accum[hit.chunk.chunk_id][0] += rrf
    fused=[]
    for score,hit in accum.values():
       fused.append(ScoredChunk(chunk=hit.chunk, score=score, retriever="rrf"))
    fused.sort(key=lambda s: s.score, reverse=True)
    return fused[:top_k]



