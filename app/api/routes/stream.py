"""流式问答接口（SSE）。"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_container
from app.container import Container
from app.models import AgentRequest
from app.services.citation import build_citations
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/v1", tags=["stream"])

# 流式模式下 LLM 要输出纯文本，不能输出 JSON（否则用户会看到花括号）
STREAM_SYSTEM = (
    "你是运维知识助手。只能依据【资料】作答，每个结论句末用 [编号] 标注来源；"
    "资料不足则如实说明并建议转人工。"
    "直接输出答案正文，不要输出 JSON、不要加代码块标记。"
)


def _sse(data: dict) -> str:
    """SSE 协议：每条消息必须是 'data: <内容>\\n\\n'。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask/stream", summary="流式问答（SSE）")
async def ask_stream(
    payload: AgentRequest,
    container: Container = Depends(get_container),
):
    llm_client = container.llm

    async def event_source():
        # 1) 开始
        yield _sse({"type": "meta", "question": payload.question})

        if llm_client is None:
            yield _sse({"type": "error", "message": "LLM 未就绪，已降级"})
            yield _sse({"type": "done"})
            return

        # ① 检索（用 container.retrieve）
        #    提示：参考 ask.py:30-31 的写法
        started = time.perf_counter()

        settings = container.settings
        top_k = payload.top_k or settings.top_k
        scope = container.knowledge.set_service_scope(payload.service)
        hits = await run_in_threadpool(container.retrieve,
                                       payload.question,
                                       top_k=top_k,
                                       service=scope)

        # ② 构造引用并发送
        citations = build_citations(hits)
        yield _sse({"type":"sources", "citations": [c.model_dump() for c in citations]})
        # ③ 构造带资料的提示词
        context = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))
        user_prompt = f"【问题】{payload.question}\n\n【资料】\n{context}"

        try:
            # 2) 逐 token 流式输出
            #    必须用 chat_stream_async：同步版在等网络时会阻塞整个事件循环，
            #    3 秒的生成时间会把其他所有并发请求一起卡住
            async for piece in llm_client.chat_stream_async(STREAM_SYSTEM, user_prompt):
                yield _sse({"type": "token", "text": piece})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            # 3) 结束（无论成功失败都发）
            yield _sse({"type": "done",
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        })

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )