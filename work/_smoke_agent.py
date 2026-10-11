from pathlib import Path

from app.config import Settings
from app.container import build_container
from app.services.agent import build_agent_graph
from app.services.tickets import TicketStore

settings = Settings(ticket_store=Path("work/state/_smoke_tickets.json"))
container = build_container(settings)
container.knowledge.load_directory()
graph=container.agent_graph

for q in ("熔断错误率阈值是多少", "支付网关挂了，帮我建工单", "今天天气怎么样"):
    state = graph.invoke({"question": q})
    # print(state["intent"], "->", state["answer"])

    print(state["intent"], "->", state["answer"], "| 工具:", state.get("tools_used", []), "| 引用数:",
          len(state.get("citations", [])))