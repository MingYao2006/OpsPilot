"""健康与就绪探针。降级状态如实上报，不掩盖。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.container import Container
from app.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="存活探针 + 实际生效的后端")
def healthz(container: Container = Depends(get_container)) -> HealthResponse:
    settings = container.settings
    llm = settings.resolve_llm()

    details = {
        "uptime_s": round(container.uptime_seconds(), 1),
        "vector_backend": container.store.health(),
        "embedder": container.embedder.health(),
        "llm": {
            "provider": llm["provider"],
            "model": llm["model"],
            "base_url": llm["base_url"],
            "ready": settings.llm_ready,
        },
        "knowledge": container.knowledge.stats(),
        "degradation_notes": container.notes,
        "startup": container.startup_summary,
    }
    return HealthResponse(
        status="degraded" if container.degraded else "ok",
        version=settings.version,
        details=details,
    )


@router.get("/readyz", response_model=HealthResponse, summary="就绪探针")
def readyz(container: Container = Depends(get_container)) -> HealthResponse:
    problems: list[str] = []

    try:
        chunks = container.knowledge.count()
    except Exception as exc:  # noqa: BLE001
        chunks = 0
        problems.append(f"向量库不可访问：{exc}")

    if chunks == 0:
        problems.append("知识库为空，请先执行 python scripts/seed.py")

    return HealthResponse(
        status="ok" if not problems else "degraded",
        version=container.settings.version,
        details={"chunks": chunks, "problems": problems},
    )
