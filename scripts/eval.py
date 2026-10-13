"""D7 评测扩题：正面 / 同义换词 / 跨文档 OR / 阴性对照 四维覆盖。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.utils.logging import configure_logging  # noqa: E402

# ---------- 评测用例 ----------
# 格式：(query, expect, category)
# expect: "01" 精确文档, "01|06" 跨文档 OR, "NONE" 阴性对照
CASES: list[tuple[str, str, str]] = [
    # === 正面：精确查询 (D2 原始 9 题) ===
    ("支付网关连接池耗尽的止血步骤是什么", "01", "正面"),
    ("核心服务的依赖关系和整体架构", "02", "正面"),
    ("P1 告警多久没响应会自动升级", "03", "正面"),
    ("灰度发布与回滚的标准作业流程", "04", "正面"),
    ("熔断的错误率阈值和窗口大小怎么配", "05", "正面"),
    ("2026-07-15 支付超时故障的根因", "06", "正面"),
    ("慢查询的判定阈值是多少秒", "07", "正面"),
    ("值班交接必须覆盖哪几项内容", "08", "正面"),
    ("一线工单处置超过多久要升级二线", "08", "正面"),
    # === 同义换词：不同措辞、相同期望 ===
    ("连接池打满了怎么恢复", "01", "换词"),
    ("发布后怎么退回上一版本", "04", "换词"),
    ("错误率超多少触发熔断", "05", "换词"),
    # === 跨文档 OR：Top-3 命中其中一篇即 pass ===
    ("支付故障从止血到复盘全流程", "01|06", "跨文档"),
    ("数据库和告警升级的关系", "03|07", "跨文档"),
    ("怎么升级给二线", "03|08", "跨文档"),
    # === 阴性对照：知识库绝对没有的内容，期望不命中 ===
    ("红烧肉怎么做才好吃", "NONE", "阴性"),
    ("帮我推荐一部科幻电影", "NONE", "阴性"),
    ("吉他初学者怎么按 F 和弦", "NONE", "阴性"),
]


def hit(source: str, expect: str) -> bool:
    """命中判定：支持精确前缀、OR 多文档、NONE（阴性永远不 hit）。"""
    if expect == "NONE":
        return False
    if "|" in expect:
        return any(source.startswith(p) for p in expect.split("|"))
    return source.startswith(expect)


def is_negative_correct(hits: list, threshold: float) -> bool:
    """阴性对照 pass：无结果 或 所有 Top-3 分数 < threshold。"""
    if not hits:
        return True
    return all(h.score < threshold for h in hits[:3])


def evaluate(container, use_rerank: bool) -> dict[str, tuple[int, int]]:
    """按 category 分组，返回 {类别: (top1通过数, top3通过数)}。"""
    raw: dict[str, list[tuple[int, int]]] = {}
    threshold = container.settings.min_relevance
    for query, expect, cat in CASES:
        hits = container.retrieve(query, top_k=3, use_rerank=use_rerank)
        if expect == "NONE":
            p1 = int(is_negative_correct(hits[:1], threshold))
            p3 = int(is_negative_correct(hits, threshold))
        else:
            p1 = int(len(hits) >= 1 and hit(hits[0].chunk.source, expect))
            p3 = int(any(hit(h.chunk.source, expect) for h in hits[:3]))
        raw.setdefault(cat, []).append((p1, p3))
    return {cat: (sum(p[0] for p in v), sum(p[1] for p in v)) for cat, v in raw.items()}


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = build_container(settings)
    container.knowledge.load_directory()

    hybrid = evaluate(container, use_rerank=False)
    ranked = evaluate(container, use_rerank=True)

    print("=" * 60)
    print(f"{'OpsPilot D7 评测':^52}")
    print("=" * 60)
    header = f"{'类别':<6} {'用例数':<6} {'模式':<12} {'Top-1':<8} {'Top-3':<8}"
    print(header)
    print("-" * 60)

    for cat in ("正面", "换词", "跨文档", "阴性"):
        total = sum(1 for _, _, c in CASES if c == cat)
        h = hybrid[cat]
        r = ranked[cat]
        print(f"{cat:<6} {total:<6} {'混合(不精排)':<12} {h[0]}/{total:<6} {h[1]}/{total}")
        print(f"{'':<6} {'':<6} {'混合+精排':<12} {r[0]}/{total:<6} {r[1]}/{total}")

    # 总计
    n = len(CASES)
    h1 = sum(v[0] for v in hybrid.values())
    h3 = sum(v[1] for v in hybrid.values())
    r1 = sum(v[0] for v in ranked.values())
    r3 = sum(v[1] for v in ranked.values())
    print("-" * 60)
    print(f"{'总计':<6} {n:<6} {'混合(不精排)':<12} {h1}/{n:<6} {h3}/{n}")
    print(f"{'':<6} {'':<6} {'混合+精排':<12} {r1}/{n:<6} {r3}/{n}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
