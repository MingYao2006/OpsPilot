"""
FastAPI 依赖注入：统一从这里取容器与子服务。
依赖注入：容器获取 + JWT 鉴权
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings
from app.container import Container
from app.services.knowledge import KnowledgeBase


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务尚未初始化完成，请稍后重试",
        )
    return container


def get_kb(container: Container = Depends(get_container)) -> KnowledgeBase:
    return container.knowledge


def get_app_settings(container: Container = Depends(get_container)) -> Settings:
    return container.settings


def _decode_hs256(token: str, secret: str) -> dict:
    """用标准库校验 HS256 JWT，避免为可选鉴权引入额外依赖。"""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Token 结构非法")

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()

    padding = "=" * (-len(signature_b64) % 4)
    actual = base64.urlsafe_b64decode(signature_b64 + padding)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("签名不匹配")

    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    expires_at = payload.get("exp")
    if expires_at is not None and time.time() > float(expires_at):
        raise ValueError("Token 已过期")
    return payload


async def require_auth(
    container: Container = Depends(get_container),
    authorization: str | None = Header(default=None),
) -> dict:
    """可选鉴权。未配置 OPSPILOT_JWT_SECRET 时放行，便于本地开发。

    返回的 claims 会用于检索前的范围收口（M4）。
    """
    secret = container.settings.jwt_secret
    if not secret:
        return {"sub": "anonymous", "scope": "all"}

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return _decode_hs256(authorization.split(" ", 1)[1].strip(), secret)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 校验失败：{exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
