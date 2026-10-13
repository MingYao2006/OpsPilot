"""JWT 校验依赖 app/api/deps.py 的测试：纯函数 _decode_hs256 四态 + require_auth 鉴权矩阵。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from app.api.deps import _decode_hs256

SECRET = "test-secret"


def _b64url(obj: dict) -> str:
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def make_token(payload: dict, secret: str = SECRET) -> str:
    header = _b64url({"alg": "HS256", "typ": "JWT"})
    body = _b64url(payload)
    signing_input = f"{header}.{body}".encode("ascii")
    # 👇 你来写：算 sig（hmac.new(...).digest() 再 urlsafe_b64encode + rstrip("=")），return f"{header}.{body}.{sig}"
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
