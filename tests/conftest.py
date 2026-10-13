"""pytest 全局配置：统一把测试环境固定为「离线可跑」的组合。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def offline_env(tmp_path_factory):
    """把环境钉死在 memory + local embedder + fallback LLM，保证测试不依赖网络与密钥。"""
    state_dir = tmp_path_factory.mktemp("opspilot-state")
    os.environ.update(
        {
            "OPSPILOT_VECTOR_BACKEND": "memory",
            "OPSPILOT_EMBEDDING_PROVIDER": "local",
            "OPSPILOT_EMBEDDING_DIM": "512",
            "OPSPILOT_LLM_PROVIDER": "fallback",
            "OPSPILOT_KNOWLEDGE_DIR": str(PROJECT_ROOT / "data" / "kb"),
            "OPSPILOT_STATE_DIR": str(state_dir),
            "OPSPILOT_TICKET_STORE": str(state_dir / "tickets.json"),
            "OPSPILOT_AUTOLOAD_KNOWLEDGE": "true",
            "OPSPILOT_LOG_LEVEL": "WARNING",
            "OPSPILOT_JWT_SECRET": "",
            # 钉死限流开关：避免宿主 shell 里 export 的 OPSPILOT_RATE_LIMIT_ENABLED 泄漏进测试把 TestClient 打成 429
            "OPSPILOT_RATE_LIMIT_ENABLED": "false",
        }
    )

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def settings(offline_env):
    from app.config import get_settings

    cfg = get_settings()
    cfg.ensure_dirs()
    return cfg


@pytest.fixture(scope="session")
def container(settings):
    from app.container import build_container

    built = build_container(settings)
    built.startup_summary = built.knowledge.load_directory()
    return built


@pytest.fixture(scope="session")
def client(offline_env):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
