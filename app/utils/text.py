"""文本处理工具：中英文混合分词、分句、稳定 ID、数值抽取。

不依赖 jieba 等外部词典，采用「ASCII 词 + CJK 单字与二元组」的混合切分，
对运维术语（如「连接池耗尽」「熔断降级」）能达到可用的检索区分度。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-z0-9_]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")
_SENT_BREAK = re.compile(r"([。！？；!?;]|\n+)")

# 高频虚词，去掉后能显著提升 BM25 的区分度
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "有", "为", "对", "从", "到",
    "个", "我们", "你们", "他们", "这", "那", "并", "被", "把", "中", "上", "下",
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "be", "it", "this", "that", "with", "as", "by", "at", "from", "if", "then",
}


def normalize(text: str) -> str:
    """NFKC 归一化 + 统一换行 + 压缩空白，保留段落结构。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """混合分词：ASCII 词 + CJK 单字与二元组，已去停用词。"""
    lowered = normalize(text).lower()
    tokens: list[str] = []
    tokens.extend(_ASCII_WORD.findall(lowered))
    for run in _CJK_RUN.findall(lowered):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [t for t in tokens if t and t not in _STOPWORDS]


def content_tokens(text: str) -> set[str]:
    """用于重合度计算的词集合（优先二元组，信息量更高）。"""
    toks = tokenize(text)
    bigrams = {t for t in toks if len(t) == 2}
    return bigrams or set(toks)


def split_sentences(text: str) -> list[str]:
    """按中英文标点与换行切句，并保留句尾分隔符。

    保留分隔符很关键：分块时句子会被直接拼接，
    若丢掉分隔符，重建出的文本会丢失段落结构，既影响可读性也影响 LLM 理解。
    """
    normalized = normalize(text)
    pieces = _SENT_BREAK.split(normalized)

    sentences: list[str] = []
    for index in range(0, len(pieces) - 1, 2):
        combined = pieces[index] + pieces[index + 1]
        # 纯换行段也要保留：它承载段落边界，丢弃后分块文本会粘连
        if combined.strip() or "\n" in pieces[index + 1]:
            sentences.append(combined)

    if len(pieces) % 2 == 1 and pieces[-1].strip():
        sentences.append(pieces[-1])

    return sentences


def stable_id(*parts: str, size: int = 16) -> str:
    """由内容派生的稳定 ID，保证重复入库幂等。"""
    digest = hashlib.sha1("\u241f".join(parts).encode("utf-8")).hexdigest()
    return digest[:size]


def extract_numbers(text: str) -> set[str]:
    """抽取数值字面量，用于答案与引用原文的数字一致性校验。"""
    found = set()
    for raw in _NUMBER.findall(normalize(text)):
        found.add(raw)
        found.add(raw.rstrip("%"))
    return {f for f in found if f}


def lexical_overlap(source: str, target: str) -> float:
    """target 有多少比例的内容词出现在 source 中（用于引用支撑度）。"""
    src = content_tokens(source)
    tgt = content_tokens(target)
    if not tgt:
        return 0.0
    return len(src & tgt) / len(tgt)


def snippet(text: str, limit: int = 160) -> str:
    """生成引用摘要，按句边界截断避免断句。"""
    text = normalize(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("。", "！", "？", "；", ".", "!", "?", ";"):
        pos = cut.rfind(sep)
        if pos >= limit // 2:
            return cut[: pos + 1]
    return cut + "…"
