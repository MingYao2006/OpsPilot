"""答案合成：检索证据 → LLM 结构化作答；LLM 离线则诚实拒答。"""
from __future__ import annotations

import json

from app.config import Settings
from app.models import AnswerPayload, ScoredChunk
from app.services.citation import VerificationResult, build_citations, verify_answer
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


def _refuse(hits: list[ScoredChunk], *, offline: bool, reason: str | None = None) -> AnswerPayload:
    citations = build_citations(hits)
    used = [c.index for c in citations][:1]
    if hits and offline:
        answer = f"检索到 {len(hits)} 条相关资料，但当前离线（未启用 LLM），无法生成可靠结论，已转人工处理。"
    elif hits and reason:
        answer = f"检索到 {len(hits)} 条相关资料，但生成的答复未通过引用校验（{reason}），已转人工复核。"
    elif hits:
        answer = f"检索到 {len(hits)} 条相关资料，已征询 LLM 但其答复连续不符合规范，为保证准确已转人工复核。"
    else:
        answer = "未检索到相关资料，且当前无法生成可靠结论，已转人工处理。"
    return AnswerPayload(answer=answer, used_citations=used, confidence=0.0, needs_human=True)

HISTORY_TURNS = 6

def format_history(history: list[str], *, max_turns: int = HISTORY_TURNS) -> str:
    """把最近 max_turns 轮历史渲染成提示词前缀；无历史返回空串（零回归）。"""
    if not history:
        return ""
    lines = "\n".join(f"- {turn}" for turn in history[-max_turns:])
    return f"【对话历史】\n{lines}\n\n"

def compose_answer(
        question: str, hits: list[ScoredChunk], llm: LLMClient | None, settings: Settings, *,
        history: list[str] | None = None,
) -> AnswerPayload:
    if  llm is None:
        return _refuse(hits,offline=True)

    user = f"{format_history(history or [])}【问题】{question}\n\n【资料】\n{_context(hits)}"
    citations = build_citations(hits)
    source_texts = [h.chunk.text for h in hits]
    last_err: Exception | None = None
    last_verdict: VerificationResult | None = None

    for _ in range(settings.llm_max_retries + 1):  # 解析失败可重试
        raw = llm.chat(SYSTEM, user)
        try:
            payload = AnswerPayload.model_validate_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:  # Pydantic 错误也是 ValueError 子类
            last_err = exc
            continue

        # 解析成功还要过引用校验：数字对不上或大面积无据可依，一律拒答。
        # 这是"绝不编造操作步骤"承诺的实际执行点，而不只是提示词里的一句话。
        if not settings.verify_citations:
            return payload

        verdict = verify_answer(
            payload.answer,
            citations,
            source_texts=source_texts,
            min_support=settings.min_citation_support,
        )
        if verdict.ok:
            return payload

        last_verdict = verdict
        log.warning("引用校验未通过，重试：%s", "；".join(verdict.reasons))
        # 把失败原因回灌给模型，给它一次改正的机会（而不是原地重试同样的问题）
        user = (
            f"{user}\n\n【上一次答复未通过校验】{'；'.join(verdict.reasons)}。"
            "请严格只依据【资料】重写，数字必须与资料完全一致，并在结论句末标注 [编号]。"
        )

    if last_verdict is not None:
        log.warning("引用校验连续未通过：%s", "；".join(last_verdict.reasons))
        return _refuse(hits, offline=False, reason="；".join(last_verdict.reasons))
    log.warning("LLM 答复连续不合契约：%s", last_err)
    return _refuse(hits,offline=False)  # 多次解析失败 → 安全落到拒答
