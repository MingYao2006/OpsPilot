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
