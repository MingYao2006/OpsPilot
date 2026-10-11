"""引用构建与校验原语。

三个层次的检查（verify 编排在 M4 接入）：
1. 引用存在性 —— 正文中出现的 [n] 必须能在 citations 中找到
2. 引用支撑度 —— 答案句子有多少内容词落在被引原文中
3. 数字一致性 —— 答案里的数字必须出现在被引原文里（运维场景最关键）
"""

from __future__ import annotations

from app.models import Citation, ScoredChunk
from app.utils.text import extract_numbers, lexical_overlap, snippet, split_sentences


def build_citations(hits: list[ScoredChunk]) -> list[Citation]:
    """把检索命中转换为带序号的引用卡片（正文以 [1][2] 形式标注）。"""
    return [
        Citation(
            index=index,
            doc_id=hit.chunk.doc_id,
            title=hit.chunk.title,
            source=hit.chunk.source,
            chunk_id=hit.chunk.chunk_id,
            snippet=snippet(hit.chunk.text),
            score=round(float(hit.score), 4),
        )
        for index, hit in enumerate(hits, start=1)
    ]


def citation_support(sentence: str, chunk_texts: list[str]) -> float:
    """句子有多少比例的内容词能在被引原文中找到（0~1）。"""
    if not chunk_texts:
        return 0.0
    return max(lexical_overlap(text, sentence) for text in chunk_texts)


def unsupported_numbers(answer: str, chunk_texts: list[str]) -> set[str]:
    """答案中出现、但被引原文里找不到的数字。"""
    allowed: set[str] = set()
    for text in chunk_texts:
        allowed |= extract_numbers(text)
    return extract_numbers(answer) - allowed


def used_citation_indexes(answer: str) -> set[int]:
    """从正文里解析出实际被引用的序号。"""
    import re

    found = set()
    for raw in re.findall(r"\[(\d{1,2})\]", answer):
        try:
            found.add(int(raw))
        except ValueError:
            continue
    return found


def sentence_support_report(answer: str, chunk_texts: list[str]) -> list[tuple[str, float]]:
    """逐句给出支撑度，便于定位无据可依的句子。"""
    return [(s, citation_support(s, chunk_texts)) for s in split_sentences(answer)]
