"""Agent 编排：LangGraph 状态图。节点只消费 C2/B/C1 成果，不携带业务逻辑。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.models import Intent, TicketIn, ScoredChunk, Citation
from app.services.answer import compose_answer, format_history
from app.services.citation import build_citations
from app.services.intent import classify_intent
from app.services.llm import LLMClient
from app.services.tickets import TicketStore
from app.utils.logging import get_logger

log = get_logger(__name__)
SearchFn = Callable[..., list[ScoredChunk]]

_TICKET_DRAFT_PROMPT = (
    "从用户原话中抽取运维工单信息。只输出一个 JSON 对象，禁止输出任何其他文字。字段必须为：\n"
    '{"title": "一句话概括故障或诉求，不超过200字", '
    '"summary": "现象与影响的详细描述", '
    '"priority": "P1、P2、P3 或 P4", '
    '"service": "涉及的服务名，无法判断填 null"}'
)

# 真 Function Calling 的工具 schema：字段严格对齐 TicketIn。
# 注意 tools 参数是 list[dict]，故即使只有一个工具也要外层套方括号。
_CREATE_TICKET_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "根据用户原话创建一张运维工单，抽取故障标题、详情、优先级与涉及服务",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "一句话概括故障或诉求，不超过200字",
                    },
                    "summary": {
                        "type": "string",
                        "description": "现象与影响的详细描述",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["P1", "P2", "P3", "P4"],
                        "description": "故障优先级；无法判断时选 P3",
                    },
                    "service": {
                        "type": "string",
                        "description": "涉及的服务名；无法判断可省略",
                    },
                },
                "required": ["title", "summary"],
            },
        },
    },
]

_TICKET_TOOL_SYSTEM = (
    "你是运维工单助手。当需要创建工单时，调用 create_ticket 工具，"
    "从用户原话中抽取标题、详情、优先级与涉及服务。"
)

_CHAT_SYSTEM = (
    "你是企业运维知识助手。用一两句口语化的中文回应用户的寒暄或闲聊，"
    "并适时提示：技术问题可直接提问（系统会检索知识库），故障可要求建工单。"
    "禁止编造任何技术结论。"
)


class AgentState(TypedDict, total=False):
    question: str
    intent: Intent
    confidence: float
    source: str
    answer: str
    ticket_id: str
    answer_confidence: float
    needs_human: bool
    citations: list[Citation]
    tools_used: list[str]
    follow_up: list[str]
    service: str | None
    top_k: int | None
    history: list[str]


def build_agent_graph(llm: LLMClient | None, tickets: TicketStore, search: SearchFn, settings: Settings):
    """装配状态图。llm 由容器注入（None = 离线，全走规则轨）。"""

    def _draft_ticket(question: str) -> tuple[TicketIn, str]:
        """返回 (建单载荷, 建单方式)。三级降级：真FC → JSON-prompt → 硬兜底，全程不静默。"""
        if llm is not None:
            # 1) 真 Function Calling
            try:
                message = llm.chat_with_tools(_TICKET_TOOL_SYSTEM, question, _CREATE_TICKET_TOOL)
                raw_args = message["tool_calls"][0]["function"]["arguments"]
                return TicketIn.model_validate_json(raw_args), "tool_call"
            except Exception as exc:
                log.warning("真 FC 建单失败，退回 JSON-prompt 抽取：%s", exc)
            # 2) JSON-prompt 抽取（老路，D4 退役为 fallback）
            try:
                raw = llm.chat(_TICKET_DRAFT_PROMPT, question)
                return TicketIn.model_validate_json(raw), "json_prompt"
            except Exception as exc:
                log.warning("JSON-prompt 抽取失败，退回硬兜底：%s", exc)
        # 3) 硬兜底
        return (
            TicketIn(title=f"工单：{question[:190]}", summary=f"用户原话：{question}", priority="P3"),
            "hard",
        )

    def classify_node(state: AgentState) -> dict:
        result = classify_intent(state["question"], llm=llm)
        return {"intent": result.intent,
                "confidence": result.confidence,
                "source": result.source}

    def retrieve_node(state: AgentState) -> dict:
        hits = search(state["question"], top_k=state.get("top_k"), service=state.get("service"))
        result = compose_answer(state["question"], hits, llm, settings, history=state.get("history"))
        return {
            "answer": result.answer,
            "answer_confidence": result.confidence,
            "needs_human": result.needs_human,
            "citations": build_citations(hits),
            "follow_up": result.follow_up
        }

    def create_ticket_node(state: AgentState) -> dict:
        payload, method= _draft_ticket(state["question"])
        ticket = tickets.create(payload)
        mark = "（兜底建单）" if method=="hard" else ""
        return {
            "answer": f"{mark}已创建工单 {ticket.ticket_id}：{ticket.title}",
            "ticket_id": ticket.ticket_id,
            "tools_used": ["create_ticket"]
        }

    def chat_node(state: AgentState) -> dict:
        if llm is not None:
            try:
                user = format_history(state.get("history") or []) + state["question"]
                return {"answer": llm.chat(_CHAT_SYSTEM, user, json_mode=False)}
            except Exception as exc:
                log.warning("闲聊回复失败，落离线模板：%s", exc)
                return {
                    "answer": "你好，我是运维知识助手。技术问题直接提问（会自动检索知识库作答），"
                              "故障请说\u201c帮我建工单\u201d。（当前离线，仅能引导，无法真正闲聊）"
                }

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("create_ticket", create_ticket_node)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "classify")

    def route_by_intent(state: AgentState) -> str:
        return state["intent"]

    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "retrieve": "retrieve",
            "create_ticket": "create_ticket",
            "chat": "chat",
        },
    )

    for name in ("retrieve", "create_ticket", "chat"):
        graph.add_edge(name, END)
    return graph.compile()
