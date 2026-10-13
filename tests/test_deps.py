"""JWT 校验依赖 app/api/deps.py 的测试：纯函数 _decode_hs256 四态 + require_auth 鉴权矩阵。"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from fastapi import HTTPException
from types import SimpleNamespace

import pytest

from app.api.deps import _decode_hs256, require_auth
from app.config import Settings

SECRET = "test-secret"


def _b64url(obj: dict) -> str:
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def make_token(payload: dict, secret: str = SECRET) -> str:
    header = _b64url({"alg": "HS256", "typ": "JWT"})
    body = _b64url(payload)
    signing_input = f"{header}.{body}".encode("ascii")
    sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    return f"{header}.{body}.{sig}"


class TestDecodeHS256:
    def test_valid_token_returns_payload(self):
        token = make_token({"sub": "alice", "exp": time.time() + 3600})
        claims = _decode_hs256(token, SECRET)
        assert claims["sub"] == "alice"

    def test_expired_token_raises(self):
        token = make_token({"sub": "x", "exp": time.time() - 1})
        with pytest.raises(ValueError):
            _decode_hs256(token, SECRET)

    def test_bad_signature_raises(self):
        # 用错误 secret 签名 → ValueError("签名不匹配")
        token = make_token(payload={"sub": "x"}, secret="wrong-secret")
        with pytest.raises(ValueError):
            _decode_hs256(token, SECRET)

    def test_malformed_token_raises(self):
        with pytest.raises(ValueError):
            _decode_hs256("a.b", SECRET)  # 只有两段 → 命中 "Token 结构非法"


class TestRequireAuth:
    def _container(self, secret: str | None):
        return SimpleNamespace(settings=Settings(jwt_secret=secret))

    def test_no_secret_allows_anonymous(self):
        claims = asyncio.run(require_auth(self._container(None), None))
        assert claims == {"sub": "anonymous", "scope": "all"}

    def test_lost_header(self):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(require_auth(self._container(SECRET), None))
        assert ei.value.status_code == 401

    def test_invalid_token(self):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(require_auth(self._container(SECRET), "Bearer a.b"))
        assert ei.value.status_code == 401

    def test_invalid_signature(self):
        token = make_token(payload={"sub": "x"}, secret="wrong-secret")
        with pytest.raises(HTTPException) as ei:
            asyncio.run(require_auth(self._container(SECRET), f"Bearer {token}"))
        assert ei.value.status_code == 401

    def test_secret(self):
        token = f"Bearer {make_token({'sub': 'bob', 'exp': time.time() + 3600})}"
        claims = asyncio.run(require_auth(self._container(SECRET), token))
        assert claims["sub"] == "bob"
