"""引用构建与校验原语。

三个层次的检查（verify 编排在 M4 接入）：
1. 引用存在性 —— 正文中出现的 [n] 必须能在 citations 中找到
2. 引用支撑度 —— 答案句子有多少内容词落在被引原文中
3. 数字一致性 —— 答案里的数字必须出现在被引原文里（运维场景最关键）
"""

from __future__ import annotations

import re

from app.models import Citation, ScoredChunk
from app.utils.text import extract_numbers, lexical_overlap, snippet, split_sentences

# 正文里的引用标记 [1] / [12]。抽取数字前必须先摘掉：
# 否则 "[1]" 里的 1 会被当成答案编造的数字，每句带引用的答案都会误报。
_CITATION_MARKER = re.compile(r"\[\d{1,2}\]")


def strip_citation_markers(text: str) -> str:
    """移除 [n] 引用标记，留下纯正文。"""
    return _CITATION_MARKER.sub(" ", text)


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
    """答案中出现、但被引原文里找不到的数字（已排除 [n] 引用标记）。"""
    allowed: set[str] = set()
    for text in chunk_texts:
        allowed |= extract_numbers(text)
    return extract_numbers(strip_citation_markers(answer)) - allowed


def used_citation_indexes(answer: str) -> set[int]:
    """从正文里解析出实际被引用的序号。"""
    return {int(raw[1:-1]) for raw in _CITATION_MARKER.findall(answer)}


def sentence_support_report(answer: str, chunk_texts: list[str]) -> list[tuple[str, float]]:
    """逐句给出支撑度，便于定位无据可依的句子。"""
    return [
        (s, citation_support(s, chunk_texts))
        for s in split_sentences(strip_citation_markers(answer))
    ]


class VerificationResult:
    """一次答案校验的完整结论；refused 为 True 时调用方应改走拒答。"""

    def __init__(
        self,
        *,
        ok: bool,
        reasons: list[str],
        fabricated_numbers: set[str],
        unsupported_sentences: list[str],
        missing_citations: set[int],
    ) -> None:
        self.ok = ok
        self.reasons = reasons
        self.fabricated_numbers = fabricated_numbers
        self.unsupported_sentences = unsupported_sentences
        self.missing_citations = missing_citations

    @property
    def refused(self) -> bool:
        return not self.ok

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"VerificationResult(ok={self.ok}, reasons={self.reasons})"


def verify_answer(
    answer: str,
    citations: list[Citation],
    *,
    source_texts: list[str] | None = None,
    min_support: float = 0.25,
    max_unsupported_ratio: float = 0.5,
) -> VerificationResult:
    """把三层检查串起来，给出一句话结论 + 可上报的原因列表。

    source_texts 传被引块的**完整原文**（与 citations 同序）。不传则退回用
    citation.snippet —— 但 snippet 截断到 160 字，落在截断之后的数字会被误判成
    编造，故调用方应尽量传全文。

    判定口径（宁松勿误杀，但编造数字零容忍）：
    - 编造数字：出现被引原文里没有的数字 → 直接不通过（运维场景最危险）
    - 引用存在性：正文引用了不存在的 [n] → 不通过
    - 支撑度：支撑度低于 min_support 的句子占比超过 max_unsupported_ratio → 不通过
    """
    reasons: list[str] = []

    cited_texts = source_texts if source_texts is not None else [c.snippet for c in citations]
    known = {c.index for c in citations}

    missing = {i for i in used_citation_indexes(answer) if i not in known}
    if missing:
        reasons.append(f"引用了不存在的编号 {sorted(missing)}")

    fabricated = unsupported_numbers(answer, cited_texts)
    if fabricated:
        reasons.append(f"出现原文中不存在的数字 {sorted(fabricated)}")

    sentences = sentence_support_report(answer, cited_texts)
    unsupported = [s for s, score in sentences if score < min_support]
    if sentences and len(unsupported) / len(sentences) > max_unsupported_ratio:
        reasons.append(f"{len(unsupported)}/{len(sentences)} 句在原文中缺乏支撑")

    return VerificationResult(
        ok=not reasons,
        reasons=reasons,
        fabricated_numbers=fabricated,
        unsupported_sentences=unsupported,
        missing_citations=missing,
    )
