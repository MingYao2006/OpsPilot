"""答案合成：检索证据 → LLM 结构化作答；LLM 离线则诚实拒答。"""
from __future__ import annotations

import json

from app.config import Settings
from app.models import AnswerPayload, ScoredChunk
from app.services.citation import build_citations
from app.services.llm import LLMClient
from app.utils.logging import get_logger

SYSTEM = (
    "你是运维知识助手。只能依据【资料】作答，每个结论句末用 [编号] 标注来源；"
    "资料不足则如实说明并置 needs_human=true，绝不编造操作步骤。"
    "严格输出 JSON：answer/used_citations/confidence/needs_human/follow_up；"
    "follow_up 必须是字符串数组（可为空数组 []），不得是单个字符串。"
)

log = get_logger(__name__)

def _context(hits: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))


def _refuse(hits: list[ScoredChunk], *, offline: bool) -> AnswerPayload:
    citations = build_citations(hits)
    used = [c.index for c in citations][:1]
    if hits and offline:
        answer = f"检索到 {len(hits)} 条相关资料，但当前离线（未启用 LLM），无法生成可靠结论，已转人工处理。"
    elif hits:
        answer = f"检索到 {len(hits)} 条相关资料，已征询 LLM 但其答复连续不符合规范，为保证准确已转人工复核。"
    else:
        answer = "未检索到相关资料，且当前无法生成可靠结论，已转人工处理。"
    return AnswerPayload(answer=answer, used_citations=used, confidence=0.0, needs_human=True)

def compose_answer(
        question: str, hits: list[ScoredChunk], llm: LLMClient | None, settings: Settings
) -> AnswerPayload:
    if  llm is None:
        return _refuse(hits,offline=True)

    user = f"【问题】{question}\n\n【资料】\n{_context(hits)}"
    last_err: Exception | None = None
    for _ in range(settings.llm_max_retries + 1):  # 解析失败可重试
        raw = llm.chat(SYSTEM, user)
        try:
            return AnswerPayload.model_validate_json(raw)

        except (json.JSONDecodeError, ValueError) as exc:  # Pydantic 错误也是 ValueError 子类
            last_err = exc
    log.warning("LLM 答复连续不合契约：%s", last_err)
    return _refuse(hits,offline=False)  # 多次解析失败 → 安全落到拒答
