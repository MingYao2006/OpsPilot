"""规则兜底轨测试：纯函数，直接调用即可，无需任何 fixture。"""

from __future__ import annotations

from app.models import IntentResult
from app.services.intent import classify_by_rules, classify_intent


class TestClassifyByRules:
    def test_ticket_keyword_wins(self):
        result = classify_by_rules("帮我创建一个工单，支付网关挂了")
        assert isinstance(result, IntentResult)  # 契约类型对
        assert result.intent == "create_ticket"  # 判对了意图
        assert result.source == "rules"  # 如实交代走的是规则轨

    def test_retrieve_keywords_wins(self):
        result = classify_by_rules("熔断错误率阈值是多少")
        assert isinstance(result, IntentResult)
        assert result.intent == "retrieve"  # 判对了意图
        assert result.source == "rules"

    def test_chat(self):
        result = classify_by_rules("今天天气怎么样")
        assert isinstance(result, IntentResult)
        assert result.intent == "chat"  # 判对了意图
        assert result.source == "rules"

    def test_empty_string_falls_to_chat(self):
        result = classify_by_rules("")  # 输入空串
        assert result.intent == "chat"  # 应该兜底到哪？
        assert result.source == "rules"

    def test_rule_confidence_is_relative_not_absolute(self):
        hit = classify_by_rules("帮我创建一个工单，支付网关挂了")
        chat = classify_by_rules("今天天气怎么样")

        assert hit.confidence > chat.confidence  # 命中态的自信程度必须高于兜底态
        assert chat.confidence < 1.0, "规则轨永远不许冒充确定性"  # 拿命拍的值不许是 1.0


class _StubLLM:
    """替身演员：不联网，只背台词；chat 返回的是"网络那头的原始文本"。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, system: str, user: str) -> str:
        return self.reply


class _BoomLLM:
    """替身二号：一开口就炸，模拟网络层异常。"""

    def chat(self, system: str, user: str) -> str:
        raise RuntimeError("连接重置")


class TestClassifyIntent:
    def test_offline_goes_rules(self):
        result = classify_intent("今天天气怎么样", llm=None)
        assert result.source == "rules"
        assert result.intent == "chat"

    def test_llm_track_success(self):
        stub = _StubLLM('{"intent": "create_ticket", "confidence": 0.9}')
        result = classify_intent("随便问点什么", llm=stub)
        assert result.source == "llm"
        assert result.intent == "create_ticket"
        assert result.confidence == 0.9

    def test_llm_garbage_falls_back(self):
        stub = _StubLLM("我觉得这肯定是闲聊")  # 不是 JSON，model_validate_json 会抛
        result = classify_intent("今天天气怎么样", llm=stub)
        assert result.source == "rules"
        assert result.intent == "chat"

    def test_llm_raises_falls_back(self):
        result = classify_intent("熔断阈值是多少", llm=_BoomLLM())
        assert result.source == "rules"
        assert result.intent == "retrieve"
