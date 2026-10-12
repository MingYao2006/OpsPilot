"""多轮 lite：format_history 纯函数 + compose_answer 历史注入（Spy 捕获 prompt）。"""

from __future__ import annotations

from app.services.answer import compose_answer, format_history


class TestFormatHistory:
    def test_empty_returns_blank(self):
        assert format_history([]) == ""

    def test_renders_recent_turns(self):
        out = format_history(["用户：你好", "助手：我是运维助手"])
        assert out.startswith("【对话历史】")
        assert "用户：你好" in out and "助手：我是运维助手" in out
        assert out.endswith("\n\n")

    def test_truncates_to_last_turns(self):
        turns = [f"第{i}轮" for i in range(1, 11)]  # 10 条
        out = format_history(turns, max_turns=6)
        assert "第10轮" in out and "第5轮" in out
        assert "第4轮" not in out  # 只留最后 6：第5..第10


class TestComposeAnswerHistory:
    _VALID = '{"answer":"收到","used_citations":[],"confidence":0.5,"needs_human":false,"follow_up":[]}'

    def test_history_injected_for_retrieve(self, settings):
        captured: dict[str, str] = {}

        class Spy:
            def chat(self, system: str, user: str, **kw) -> str:
                captured["user"] = user
                return TestComposeAnswerHistory._VALID

        compose_answer("连接池怎么查", [], Spy(), settings, history=["上一轮"])
        assert captured["user"].startswith("【对话历史】")
        assert "上一轮" in captured["user"]
        assert "【问题】连接池怎么查" in captured["user"]

    def test_no_history_keeps_original_shape(self, settings):
        captured: dict[str, str] = {}

        class Spy:
            def chat(self, system: str, user: str, **kw) -> str:
                captured["user"] = user
                return TestComposeAnswerHistory._VALID

        compose_answer("连接池怎么查", [], Spy(), settings)
        assert captured["user"].startswith("【问题】")
        assert "【对话历史】" not in captured["user"]