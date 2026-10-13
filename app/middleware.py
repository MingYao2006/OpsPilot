"""请求上下文中间件：贯穿全链路的 request_id、耗时统计与统一异常日志。"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.logging import get_logger, set_request_id

log = get_logger("opspilot.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = set_request_id(request.headers.get("x-request-id"))
        # 计时
        started = time.perf_counter()

        # 尝试获取请求处理结果
        try:
            response = await call_next(request)
            # 获取请求处理结果
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.exception(
                "未处理异常 %s %s (%.1fms)", request.method, request.url.path, elapsed_ms
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{elapsed_ms:.1f}"
        log.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

def check_rate_limit(now:float, hits:list[float], limit:int, window_s:float)->tuple[bool, float, list[float]]:
        """滑动窗口限流判定（纯函数：不读时钟、不改入参）。

        now:      float，当前时间戳（秒）
        hits:     该键最近的请求时间戳列表（未清理，可能含窗口外的旧值）
        limit:    窗口内允许的最大请求数
        window_s: 窗口长度（秒）

        返回 (allowed, retry_after, kept)：
          allowed     —— 本次是否放行
          retry_after —— 拒绝时建议等待秒数（放行时为 0.0）
          kept        —— 判定后应写回的新 hits（已清理 + 视情况追加本次）
        """
        recent = [t for t in hits if t > now - window_s]
        if len(recent) >= limit:
            retry_after = max(0.0, window_s - (now - recent[0]))
            kept = recent
            return False, retry_after, kept
        else:
            return True, 0.0, recent + [now]
