"""多轮 lite：format_history 纯函数 + compose_answer 历史注入（Spy 捕获 prompt）。

另含引用校验接线测试：compose_answer 必须真的对 LLM 答复做数字/支撑度校验，
而不只是把"绝不编造"写在提示词里。
"""

from __future__ import annotations

from app.config import Settings
from app.models import Chunk, ScoredChunk
from app.services.answer import compose_answer, format_history


def _hit(text: str, chunk_id: str = "c1") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=text, title="熔断手册", source="kb/05.md"),
        score=0.5,
        retriever="vector",
    )


def _payload(answer: str) -> str:
    import json

    return json.dumps(
        {
            "answer": answer,
            "used_citations": [1],
            "confidence": 0.8,
            "needs_human": False,
            "follow_up": [],
        },
        ensure_ascii=False,
    )


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


class _ScriptedLLM:
    """按脚本依次返回答复，并记录每次收到的 prompt。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, system: str, user: str, **kw) -> str:
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else self.replies_default()

    @staticmethod
    def replies_default() -> str:
        return _payload("兜底答复 [1]。")


_SRC = "熔断器在错误率超过 50% 时打开，滑动窗口为 10 秒。"


class TestComposeAnswerVerification:
    """接线测试：校验必须真的执行，且失败时拒答而不是照原样返回。"""

    def test_grounded_answer_is_returned(self, settings):
        llm = _ScriptedLLM([_payload("熔断器在错误率超过 50% 时打开 [1]。")])
        result = compose_answer("熔断阈值", [_hit(_SRC)], llm, settings)
        assert result.needs_human is False
        assert "50%" in result.answer
        assert len(llm.prompts) == 1, "有据的答复不该触发重试"

    def test_fabricated_number_triggers_retry_then_refuses(self, settings):
        """LLM 连续编造阈值 → 必须拒答，绝不能把 80% 当结论返回。"""
        bad = _payload("熔断器在错误率超过 80% 时打开 [1]。")
        llm = _ScriptedLLM([bad, bad, bad])
        result = compose_answer("熔断阈值", [_hit(_SRC)], llm, settings)

        assert result.needs_human is True
        assert result.confidence == 0.0
        # 编造的结论句不得作为答复返回（拒答文案里会点出是哪个数字可疑，供人工核对）
        assert "超过 80% 时打开" not in result.answer
        assert len(llm.prompts) == settings.llm_max_retries + 1

    def test_retry_prompt_carries_failure_reason(self, settings):
        """重试不是原地重发：要把失败原因回灌，给模型改正的机会。"""
        bad = _payload("熔断器在错误率超过 80% 时打开 [1]。")
        good = _payload("熔断器在错误率超过 50% 时打开 [1]。")
        llm = _ScriptedLLM([bad, good])
        result = compose_answer("熔断阈值", [_hit(_SRC)], llm, settings)

        assert result.needs_human is False
        assert "50%" in result.answer
        assert "未通过校验" in llm.prompts[1], "第二次 prompt 应带上失败原因"
        assert len(llm.prompts) == 2

    def test_verification_can_be_disabled(self):
        """关掉开关后同样的编造答复会被直接放行 —— 证明校验确实来自这个开关。"""
        off = Settings(verify_citations=False)
        llm = _ScriptedLLM([_payload("熔断器在错误率超过 80% 时打开 [1]。")])
        result = compose_answer("熔断阈值", [_hit(_SRC)], llm, off)
        assert "80%" in result.answer
        assert len(llm.prompts) == 1

    def test_missing_citation_index_refused(self, settings):
        llm = _ScriptedLLM([_payload("熔断器在错误率超过 50% 时打开 [9]。")] * 3)
        result = compose_answer("熔断阈值", [_hit(_SRC)], llm, settings)
        assert result.needs_human is True
        assert result.confidence == 0.0
        # 原答复不得被当成结论返回（拒答文案会引用该编号说明原因）
        assert "时打开 [9]" not in result.answer

    def test_refusal_message_states_reason(self, settings):
        llm = _ScriptedLLM([_payload("熔断器在错误率超过 80% 时打开 [1]。")] * 3)
        result = compose_answer("熔断阈值", [_hit(_SRC)], llm, settings)
        assert "引用校验" in result.answer
        assert "人工" in result.answer