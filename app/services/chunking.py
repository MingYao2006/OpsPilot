"""句子感知的递归分块：按语义边界打包，允许带重叠的滑动窗口。"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import Chunk, DocumentIn
from app.utils.text import normalize, split_sentences, stable_id


def _hard_split(text: str, size: int) -> list[str]:
    """单句超长时按字符硬切，保证不产生超长块。"""
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def _tail_overlap(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """取尾部若干句作为下一块的开头，实现重叠。"""
    tail: list[str] = []
    length = 0
    for sent in reversed(sentences):
        if length + len(sent) > overlap:
            break
        tail.insert(0, sent)
        length += len(sent)
    return tail, length


def pack_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    """贪心打包：块内长度不超过 size，块间重叠不超过 overlap。"""
    if size <= 0:
        raise ValueError("chunk_size 必须为正数")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < size")

    blocks: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        pieces = _hard_split(sentence, size) if len(sentence) > size else [sentence]
        for piece in pieces:
            if current and current_len + len(piece) > size:
                blocks.append(current)
                current, current_len = _tail_overlap(current, overlap)
            current.append(piece)
            current_len += len(piece)

    if current:
        blocks.append(current)
    return ["".join(b).strip() for b in blocks if "".join(b).strip()]


def document_id_of(doc: DocumentIn) -> str:
    return doc.doc_id or stable_id("doc", doc.title, doc.source)


def chunk_document(doc: DocumentIn, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """把一篇文档切分为若干 Chunk，chunk_id 由内容派生（重复入库幂等）。"""
    doc_id = document_id_of(doc)
    text = normalize(doc.text)
    sentences = split_sentences(text) or [text]
    blocks = pack_sentences(sentences, chunk_size, chunk_overlap)

    updated_at = doc.updated_at or datetime.now(timezone.utc)
    chunks: list[Chunk] = []
    for ordinal, block in enumerate(blocks):
        chunks.append(
            Chunk(
                chunk_id=stable_id(doc_id, str(ordinal), block),
                doc_id=doc_id,
                text=block,
                title=doc.title,
                source=doc.source,
                service=doc.service,
                tags=list(doc.tags),
                ordinal=ordinal,
                updated_at=updated_at,
            )
        )
    return chunks
