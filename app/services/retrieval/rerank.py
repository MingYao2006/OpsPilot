"""精排：对 RRF 粗排结果做二次排序。

原实现只用「查询词覆盖率」单信号重排，实测在真实语义向量下是**负增益**
（Top-1 23→21、MRR 0.979→0.938）：覆盖率与 BM25 同属词面信号，
在已有语义向量时是重复信息，且会系统性偏爱「含更多查询词」的长块，
把「短而准」的块挤到后面。

现改为**归一化加权融合**：覆盖率与 RRF 分各自归一到 [0,1] 后加权求和，
保留粗排的共识信息，只让覆盖率做微调。权重可配（rerank_cov_weight），
设为 0 即等价于不精排。
"""
from __future__ import annotations

from app.models import ScoredChunk
from app.utils.text import lexical_overlap


def _normalize(scores: list[float]) -> list[float]:
    """min-max 归一到 [0,1]；全部相等时返回全 0（无区分度，不参与排序）。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    *,
    top_n: int,
    cov_weight: float = 0.35,
) -> list[ScoredChunk]:
    """RRF 粗排榜 → 覆盖率与 RRF 加权融合 → 取前 top_n。retriever 标记 "rerank"。

    cov_weight: 覆盖率所占权重（0~1）。0 = 纯 RRF（等价于不精排），
                1 = 纯覆盖率（旧行为）。默认 0.35 为实测较优值。
    """
    if not candidates:
        return []

    cov = [lexical_overlap(hit.chunk.text, query) for hit in candidates]
    rrf = [hit.score for hit in candidates]
    cov_n = _normalize(cov)
    rrf_n = _normalize(rrf)

    w = max(0.0, min(1.0, cov_weight))
    scored = [
        (w * c + (1.0 - w) * r, c, hit)
        for c, r, hit in zip(cov_n, rrf_n, candidates)
    ]
    # 主键融合分，次键原始 RRF（打破平局时仍尊重粗排共识）
    scored.sort(key=lambda p: (p[0], p[2].score), reverse=True)

    return [
        ScoredChunk(chunk=hit.chunk, score=round(fused, 6), retriever="rerank")
        for fused, _c, hit in scored[:top_n]
    ]
