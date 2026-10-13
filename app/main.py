"""OpsPilot 服务入口。

启动时一次性装配容器并加载知识库；未配置任何外部依赖也能跑通（自动降级）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import ask
from app.api.routes import agent
from app.api.routes import health, ingest, query,tickets
from app.config import get_settings
from app.container import build_container
from app.middleware import RequestContextMiddleware, RateLimitMiddleware
from app.utils.logging import configure_logging, get_logger

log = get_logger("opspilot.main")

DESCRIPTION = """
企业内部运维知识 Agent。

- **/v1/query** 纯检索，不调用 LLM，用于单独衡量检索质量
- **/v1/documents** 文档入库与管理
- **/healthz** 会如实上报当前实际生效的向量库、Embedding 与 LLM，以及所有降级提示
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    container = build_container(settings)
    if settings.autoload_knowledge:
        container.startup_summary = container.knowledge.load_directory()

    app.state.container = container

    log.info(
        "OpsPilot 就绪 | 向量库=%s | Embedding=%s | LLM=%s(%s, ready=%s) | 知识块=%d",
        container.store.name,
        container.embedder.name,
        settings.llm_provider,
        settings.llm_model,
        settings.llm_ready,
        container.knowledge.count(),
    )
    for note in container.notes:
        log.warning("降级提示：%s", note)

    try:
        yield
    finally:
        log.info("OpsPilot 已停止")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    # 请求上下文
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(RequestContextMiddleware)
    # 设置跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(ask.router)
    app.include_router(agent.router)
    app.include_router(tickets.router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"type": "validation_error", "detail": exc.errors()},
        )

    return app


app = create_app()
