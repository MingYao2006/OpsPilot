"""日志与请求上下文。"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid



# 日志与请求上下文工具模块,核心目标是:让每条日志都自动带上"请求 ID"


# 用contextvars实现保存请求ID
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


# 设置请求ID
def set_request_id(value: str | None = None) -> str:
    rid = value or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


# 获取请求ID

def get_request_id() -> str:
    return _request_id.get()


# 日志过滤器，将请求ID添加到日志记录中

class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


# 配置日志

def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# 获取日志记录器

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
