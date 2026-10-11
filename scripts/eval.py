"""M2 混合检索评测：对打 M1 基线 Top-1 7/9、Top-3 9/9。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.utils.logging import configure_logging  # noqa: E402

CASES = [
    ("支付网关连接池耗尽的止血步骤是什么", "01"),
    ("核心服务的依赖关系和整体架构", "02"),
    ("P1 告警多久没响应会自动升级", "03"),
    ("灰度发布与回滚的标准作业流程", "04"),
    ("熔断的错误率阈值和窗口大小怎么配", "05"),
    ("2026-07-15 支付超时故障的根因", "06"),
    ("慢查询的判定阈值是多少秒", "07"),
    ("值班交接必须覆盖哪几项内容", "08"),
    ("一线工单处置超过多久要升级二线", "08"),
]


def hit(source: str, expect: str) -> bool:
    """命中判定：文档前缀必须以 source 开头（避免日期串号）。"""
    return source.startswith(expect)



def evaluate(container, use_rerank: bool) -> tuple[int, int]:
    top1 = 0
    top3 = 0
    for query, expect in CASES:
        hits = container.retrieve(query, top_k=3, use_rerank=use_rerank)
        top1 += len(hits) >= 1 and hit(hits[0].chunk.source, expect)
        top3 += any(hit(h.chunk.source, expect) for h in hits[:3])
    return top1, top3


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = build_container(settings)
    container.knowledge.load_directory()  # 确保库非空
    hybrid = evaluate(container, use_rerank=False)  # 纯混合：向量+BM25+RRF
    ranked = evaluate(container, use_rerank=True)  # 混合 + 精排
    n = len(CASES)
    print("-" * 50)
    print(f"M1 纯向量(基线)     : Top-1 7/{n}  Top-3 9/{n}")
    print(f"M2 混合(不精排)     : Top-1 {hybrid[0]}/{n}  Top-3 {hybrid[1]}/{n}")
    print(f"M2 混合+精排        : Top-1 {ranked[0]}/{n}  Top-3 {ranked[1]}/{n}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
