"""中间件测试：纯函数 check_rate_limit 四态 + RateLimitMiddleware 经 TestClient 验 429。"""
from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import RateLimitMiddleware, check_rate_limit


class TestCheckRateLimit:
    def test_first_request_allowed(self):
        allowed, retry, kept = check_rate_limit(100, [], 3, 10)
        assert allowed is True
        assert kept == [100]  # 放行才追加本次

    def test_at_limit_denied(self):
        allowed, retry, kept = check_rate_limit(106, [100, 103, 105], 3, 10)
        assert allowed is False
        assert retry == 4  # 等到 110 最老那条滑出
        assert kept == [100, 103, 105]  # 拒绝：原样，不追加也不删

    def test_window_prunes_old(self):
        # 100 已滑出窗口 → recent=[103,105] → 放行，kept 追加 110
        allowed, retry, kept = check_rate_limit(110, [100, 103, 105], 3, 10)
        assert allowed is True
        assert kept == [103, 105, 110]
        assert retry == 0.0

    def test_retry_after_not_negative(self):
        # 构造一个边界，确认 max(0.0,...) 兜住（now 恰好让窗口内无过期）
        allowed, retry, kept = check_rate_limit(107, [100, 103, 105], 3, 10)
        assert allowed is False
        assert retry == 3
        assert kept == [100, 103, 105]


def _client(enabled=True, limit=2, window=60):
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        settings=SimpleNamespace(rate_limit_enabled=enabled, rate_limit=limit, rate_limit_window=window),
    )

    @app.get("/v1/ping")
    def ping():
        return {"ok": True}

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return TestClient(app)


# RateLimitMiddleware 走 TestClient
class TestRateLimitMiddleware:
    def test_trips_limit_returns_429(self):
        c = _client(limit=2)
        codes = [c.get("/v1/ping").status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        # 再来一发确认 429 带 Retry-After 头
        r = c.get("/v1/ping")
        assert "retry-after" in {k.lower() for k in r.headers}

    def test_disabled_passes_through(self):
        c = _client(enabled=False, limit=1)
        codes = [c.get("/v1/ping").status_code for _ in range(5)]
        assert codes == [200] * 5  # 关开关 → 完全不拦



    def test_healthz_exempt(self):
        c = _client(limit=1)
        codes = [c.get("/healthz").status_code for _ in range(5)]
        assert codes == [200] * 5  # 探针豁免，永远放行

