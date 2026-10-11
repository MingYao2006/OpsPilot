"""意图分类双轨：LLM 主轨锦上添花，规则轨保底；合流函数永远有结果。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import Intent, IntentResult
from app.services.llm import LLMClient
from app.utils.logging import get_logger

log = get_logger(__name__)

_TICKET_KEYWORDS: tuple[str, ...] = ("工单", "报障", "建单", "立案", "帮我创建")
_RETRIEVE_MARKERS: tuple[str, ...] = (
    "如何",
    "为什么",
    "是什么",
    "多少",
    "哪些",
    "阈值",
    "流程",
    "步骤",
    "手册",
    "配置",
    "排查",
    "告警",
    "复盘",
    "交接",
    "怎么处理",
    "依赖",
)
_INTENT_SYSTEM_PROMPT = (
    "你是企业运维系统的意图分类器。把用户问题分到三类之一：\n"
    "- retrieve：询问运维知识、流程、原因、配置、阈值（例：熔断错误率阈值是多少？）\n"
    "- create_ticket：要求建工单、报障、立案（例：支付网关挂了，帮我建个工单）\n"
    "- chat：与运维无关的闲聊（例：今天天气怎么样）\n"
    "只输出一个 JSON 对象，禁止输出任何其他文字。字段必须为：\n"
    '{"intent": "retrieve 或 create_ticket 或 chat", "confidence": 0.0 到 1.0 之间的小数}'
)

# 决策 3：规则轨只能给启发值，不许冒充确定性（严禁 1.0）
_CONFIDENCE_HIT = 0.7
_CONFIDENCE_CHAT = 0.2


def _normalize(text: str) -> str:
    return text.lower().strip()


def classify_by_rules(question: str) -> IntentResult:
    text = _normalize(question)
    # 决策 1：显式动作指令（建单）是最强信号，先查；泛疑问词后置
    if any(keyword in text for keyword in _TICKET_KEYWORDS):
        intent: Intent = "create_ticket"
        confidence = _CONFIDENCE_HIT
    elif any(keyword in text for keyword in _RETRIEVE_MARKERS):
        intent: Intent = "retrieve"
        confidence = _CONFIDENCE_HIT
    else:
        # 决策 3：兜底 = 没找到任何信号，理应是全函数最不自信的分支
        intent: Intent = "chat"
        confidence = _CONFIDENCE_CHAT
    return IntentResult(intent=intent, confidence=confidence, source="rules")


class _LLMIntentPayload(BaseModel):
    """LLM 内部子契约：模型只交代它该交代的，source 由流水线盖章。"""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)


def classify_by_llm(question: str, llm: LLMClient) -> IntentResult:
    raw = llm.chat(_INTENT_SYSTEM_PROMPT, question)
    payload = _LLMIntentPayload.model_validate_json(raw)
    return IntentResult(
        intent=payload.intent,
        confidence=payload.confidence,
        source="llm",
    )


def classify_intent(question: str, *, llm: LLMClient | None = None) -> IntentResult:
    if llm is None:
        return classify_by_rules(question)
    try:
        return classify_by_llm(question, llm)
    except Exception as exc:
        log.warning("LLM 意图分类失败，落回规则轨：%s", exc)
        return classify_by_rules(question)

