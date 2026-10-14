"""D7 评测：正面 / 同义换词 / 跨文档 OR / 阴性对照 四维覆盖。

评测口径的一个坑（本文件已修）：
混合检索的输出分是 RRF 倒数排名分，理论上限 = 路数 /(k+1) = 2/61 ≈ 0.0328，
而 min_relevance 是按余弦相似度量纲定的 0.06。拿 0.06 去卡 0.0328 的分，
`all(score < 0.06)` 恒为 True —— 阴性对照永远通过，等于没有刹车。

因此阴性对照改用**量纲无关**的判据：查询的内容词被命中块覆盖的比例
（lexical_overlap，0~1，与检索器无关）。实测正面用例 ≥ 0.22、域外阴性 0.000，
阈值 0.15 落在两者之间且可被真正触发。

另一个坑：评测必须可复现。默认把后端钉在 memory + local + fallback，
不受本机 .env（可能指向 pgvector / 远端 embedding）影响；
需要按真实配置评测时显式加 --live。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 必须在 import app.config 之前钉住环境：get_settings() 会读 .env 与环境变量
if "--live" not in sys.argv:
    os.environ.update(
        {
            "OPSPILOT_VECTOR_BACKEND": "memory",
            "OPSPILOT_EMBEDDING_PROVIDER": "local",
            "OPSPILOT_EMBEDDING_DIM": "512",
            "OPSPILOT_LLM_PROVIDER": "fallback",
            "OPSPILOT_KNOWLEDGE_DIR": str(PROJECT_ROOT / "data" / "kb"),
            "OPSPILOT_LOG_LEVEL": "ERROR",
        }
    )

from app.config import get_settings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.utils.logging import configure_logging  # noqa: E402
from app.utils.text import lexical_overlap  # noqa: E402

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
    # === 扩充语料后的新用例（09-14 与 01-08 存在术语重叠，考区分度）===
    ("Redis 缓存击穿的防护手段有哪些", "09", "正面"),
    ("Kafka 消费积压多少算 P1", "10", "正面"),
    ("liveness 和 readiness 探针的区别", "11", "正面"),
    ("日志里必须带哪些追踪字段", "12", "正面"),
    ("压测时 P99 多少算不达标", "13", "正面"),
    ("JWT 签名密钥轮换要注意什么", "14", "正面"),
    ("热 key 过期把数据库打挂了怎么办", "09", "换词"),
    ("消息重复消费怎么做幂等", "10", "换词"),
    ("离职员工的账号多久要回收", "14", "换词"),
    # === 阴性对照：知识库绝对没有的内容，期望不命中 ===
    ("红烧肉怎么做才好吃", "NONE", "阴性"),
    ("帮我推荐一部科幻电影", "NONE", "阴性"),
    ("吉他初学者怎么按 F 和弦", "NONE", "阴性"),
    ("怎么给猫剪指甲", "NONE", "阴性"),
    ("推荐几个适合周末爬山的路线", "NONE", "阴性"),
    ("明天上海会下雨吗", "NONE", "阴性"),
]

# 阴性对照阈值：查询内容词被命中块覆盖的比例上限。
# 标定依据（固定离线配置，29 块语料）：**检索正确的**阳性用例 Top-3 覆盖率最低 0.222，
# 域外阴性最高 0.000；取 0.11 落在空隙中部，两侧余量都足够。
# 注意：自检只统计"确实命中期望文档"的用例 —— 检索本就跑错的用例覆盖率天然低，
# 把它算进来会让阈值被迫压到 0，反而失去区分力。
NEGATIVE_OVERLAP_CEILING = 0.11


def hit(source: str, expect: str) -> bool:
    """命中判定：支持精确前缀、OR 多文档、NONE（阴性永远不 hit）。"""
    if expect == "NONE":
        return False
    if "|" in expect:
        return any(source.startswith(p) for p in expect.split("|"))
    return source.startswith(expect)


def query_overlap(question: str, hit_obj) -> float:
    """查询的内容词有多少落在该命中块里（0~1，与检索器量纲无关）。"""
    return lexical_overlap(hit_obj.chunk.text, question)


def is_negative_correct(hits: list, question: str, ceiling: float = NEGATIVE_OVERLAP_CEILING) -> bool:
    """阴性对照 pass：无结果，或所有 Top-3 的查询覆盖率都低于 ceiling。"""
    if not hits:
        return True
    return all(query_overlap(question, h) < ceiling for h in hits[:3])


def reciprocal_rank(hits: list, expect: str) -> float:
    """第一个命中期望文档的名次倒数；没命中为 0。"""
    for rank, h in enumerate(hits, start=1):
        if hit(h.chunk.source, expect):
            return 1.0 / rank
    return 0.0


def rrf_ceiling(settings) -> float:
    """两路 RRF 融合分的理论上限，用于自检阈值是否落在可达区间。"""
    return 2.0 / (settings.rrf_k + 1)


def evaluate(container, use_rerank: bool) -> dict[str, dict]:
    """按 category 分组，返回每类的通过数与 MRR。"""
    raw: dict[str, list[tuple[int, int, float]]] = {}
    for query, expect, cat in CASES:
        hits = container.retrieve(query, top_k=3, use_rerank=use_rerank)
        if expect == "NONE":
            ok = is_negative_correct(hits, query)
            p1 = p3 = int(ok)
            rr = 0.0
        else:
            p1 = int(len(hits) >= 1 and hit(hits[0].chunk.source, expect))
            p3 = int(any(hit(h.chunk.source, expect) for h in hits[:3]))
            rr = reciprocal_rank(hits, expect)
        raw.setdefault(cat, []).append((p1, p3, rr))

    result: dict[str, dict] = {}
    for cat, rows in raw.items():
        n = len(rows)
        result[cat] = {
            "n": n,
            "top1": sum(r[0] for r in rows),
            "top3": sum(r[1] for r in rows),
            "mrr": sum(r[2] for r in rows) / n if n else 0.0,
        }
    return result


def sanity_check(settings) -> list[str]:
    """自检：阴性判据必须真的能把阴性与阳性分开，否则这个"刹车"是空的。

    这正是本文件修掉的那个 bug —— 阈值 0.06 配上混合检索分数的上限 0.0328，
    `all(score < 0.06)` 恒为真，阴性对照永远通过。
    注意：min_relevance 本身没写错，它是**向量阶段**（余弦，0~1）的阈值，
    container.retrieve 里只用来过滤 vec_hits；错的是拿它去卡融合后的 RRF 分。
    """
    problems: list[str] = []
    ceiling = rrf_ceiling(settings)
    if settings.min_relevance >= ceiling:
        # 记录该量纲差异，提醒不要把两者混用；这本身不是配置错误。
        print(
            f"[提示] min_relevance={settings.min_relevance} 是向量阶段的余弦阈值，"
            f"高于两路 RRF 上限 {ceiling:.4f}；请勿用它过滤融合后的分数。"
        )
    if not (0 < NEGATIVE_OVERLAP_CEILING < 1):
        problems.append(f"阴性覆盖率阈值 {NEGATIVE_OVERLAP_CEILING} 不在 (0,1) 内")
    return problems


def separability_report(container, use_rerank: bool) -> tuple[float, float]:
    """返回 (阳性用例的最低覆盖率, 阴性用例的最高覆盖率)。

    阳性只统计**确实命中期望文档**的用例：检索本身跑错的用例覆盖率天然偏低，
    它反映的是检索缺陷而非判据缺陷，混进来会逼着阈值一路降到 0。

    两者之间必须存在空隙，且 NEGATIVE_OVERLAP_CEILING 落在空隙里，
    阴性对照才算真的能区分 —— 否则它只是在恒真地"通过"。
    """
    pos_min = 1.0
    neg_max = 0.0
    for query, expect, _cat in CASES:
        hits = container.retrieve(query, top_k=3, use_rerank=use_rerank)
        if not hits:
            continue
        top_overlap = max(query_overlap(query, h) for h in hits[:3])
        if expect == "NONE":
            neg_max = max(neg_max, top_overlap)
        elif any(hit(h.chunk.source, expect) for h in hits[:3]):
            pos_min = min(pos_min, top_overlap)
    return pos_min, neg_max


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = build_container(settings)
    container.knowledge.load_directory()

    problems = sanity_check(settings)
    hybrid = evaluate(container, use_rerank=False)
    ranked = evaluate(container, use_rerank=True)

    line = "=" * 72
    print(line)
    print(f"{'OpsPilot 检索评测':^64}")
    print(line)
    live = "--live" in sys.argv
    print(f"后端 {settings.vector_backend} / Embedding {settings.embedding_provider}"
          f"(dim={settings.embedding_dim}) / {'--live 真实配置' if live else '固定离线配置'}")
    print(f"知识块 {container.knowledge.count()} 个 | 用例 {len(CASES)} 条 | "
          f"阴性覆盖率阈值 {NEGATIVE_OVERLAP_CEILING} | RRF 上限 {rrf_ceiling(settings):.4f}")
    print(line)
    print(f"{'类别':<8}{'用例':<6}{'模式':<14}{'Top-1':<10}{'Top-3':<10}{'MRR':<8}")
    print("-" * 72)

    for cat in ("正面", "换词", "跨文档", "阴性"):
        h = hybrid[cat]
        r = ranked[cat]
        print(f"{cat:<8}{h['n']:<6}{'混合(不精排)':<14}"
              f"{h['top1']}/{h['n']:<8}{h['top3']}/{h['n']:<8}{h['mrr']:.3f}")
        print(f"{'':<8}{'':<6}{'混合+精排':<14}"
              f"{r['top1']}/{r['n']:<8}{r['top3']}/{r['n']:<8}{r['mrr']:.3f}")

    n = len(CASES)
    h1 = sum(v["top1"] for v in hybrid.values())
    h3 = sum(v["top3"] for v in hybrid.values())
    r1 = sum(v["top1"] for v in ranked.values())
    r3 = sum(v["top3"] for v in ranked.values())
    hm = sum(v["mrr"] * v["n"] for v in hybrid.values()) / n
    rm = sum(v["mrr"] * v["n"] for v in ranked.values()) / n

    print("-" * 72)
    print(f"{'总计':<8}{n:<6}{'混合(不精排)':<14}{h1}/{n:<8}{h3}/{n:<8}{hm:.3f}")
    print(f"{'':<8}{'':<6}{'混合+精排':<14}{r1}/{n:<8}{r3}/{n:<8}{rm:.3f}")
    print(line)

    # 阴性对照的"刹车"自检：阳性最低覆盖率与阴性最高覆盖率之间必须有空隙，
    # 且阈值落在空隙内。否则阴性全过只是因为判据恒真，不代表检索真的没召回。
    pos_min, neg_max = separability_report(container, use_rerank=False)
    print(f"\n阴性对照自检：阳性最低覆盖率 {pos_min:.3f} | 阴性最高覆盖率 {neg_max:.3f} "
          f"| 阈值 {NEGATIVE_OVERLAP_CEILING}")
    if pos_min <= NEGATIVE_OVERLAP_CEILING:
        problems.append(
            f"阈值 {NEGATIVE_OVERLAP_CEILING} 不低于阳性最低覆盖率 {pos_min:.3f}，会误杀正面用例"
        )
    if neg_max >= NEGATIVE_OVERLAP_CEILING:
        problems.append(
            f"阈值 {NEGATIVE_OVERLAP_CEILING} 不高于阴性最高覆盖率 {neg_max:.3f}，"
            "阴性对照无法区分（等同恒真）"
        )
    if not problems:
        print(f"  -> 空隙 ({neg_max:.3f}, {pos_min:.3f}) 有效，阴性判据可被真正触发。")

    if problems:
        print("\n[自检未通过]")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("自检通过：阴性判据落在阳性/阴性的可分离区间内。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
