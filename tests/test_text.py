"""文本工具测试：中英混合分词、分句、数值抽取、重合度。"""

from __future__ import annotations

import pytest

from app.utils.text import (
    content_tokens,
    extract_numbers,
    lexical_overlap,
    normalize,
    snippet,
    split_sentences,
    stable_id,
    tokenize,
)


class TestTokenize:
    def test_cjk_produces_unigrams_and_bigrams(self):
        tokens = tokenize("连接池")
        assert "连" in tokens and "接" in tokens and "池" in tokens
        assert "连接" in tokens and "接池" in tokens

    def test_ascii_words_are_lowercased(self):
        assert "hikaricp" in tokenize("HikariCP maximumPoolSize")

    def test_stopwords_removed(self):
        assert "的" not in tokenize("支付的连接池")

    def test_splits_mixed_script(self):
        tokens = tokenize("HikariCP 连接池 maxActive=50")
        assert "hikaricp" in tokens
        assert "50" in tokens
        assert "连接" in tokens

    def test_empty_input(self):
        assert tokenize("") == []


class TestSplitSentences:
    def test_round_trip_preserves_every_character(self):
        """分句必须是无损的：拼回去要等于原文，否则分块会破坏原文结构。"""
        samples = [
            "第一句。第二句。",
            "阈值是 3 秒。\n超时即告警。",
            "- 第一项\n- 第二项\n",
            "# 标题\n\n正文内容。\n\n## 二级标题\n\n- `hikaricp` 大于 5",
            "没有标点的单句",
        ]
        for text in samples:
            assert "".join(split_sentences(text)) == normalize(text)

    def test_splits_on_cjk_period(self):
        assert split_sentences("第一句。第二句。") == ["第一句。", "第二句。"]

    def test_preserves_blank_line_boundary(self):
        sentences = split_sentences("上文。\n\n下文。")
        assert any("\n" in s for s in sentences)

    def test_empty_input(self):
        assert split_sentences("") == []


class TestExtractNumbers:
    def test_extracts_plain_and_percent(self):
        found = extract_numbers("错误率 50% 持续 3 分钟")
        assert "50" in found and "50%" in found and "3" in found

    def test_extracts_decimals(self):
        assert "0.1" in extract_numbers("错误率 < 0.1%")

    def test_no_numbers(self):
        assert extract_numbers("没有数字") == set()


class TestLexicalOverlap:
    def test_full_overlap(self):
        assert lexical_overlap("连接池耗尽", "连接池耗尽") == pytest.approx(1.0)

    def test_partial_overlap(self):
        # 源文本覆盖了目标句子 8 个二元组中的 6 个
        score = lexical_overlap("数据库连接池耗尽导致超时", "连接池耗尽导致雪崩")
        assert 0.5 < score < 1.0

    def test_subset_scores_full(self):
        # 目标句子的所有词都在源文本中，重合度应为 1.0
        assert lexical_overlap("数据库连接池耗尽导致超时", "连接池耗尽") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert lexical_overlap("连接池", "年会地点") == 0.0

    def test_empty_target(self):
        assert lexical_overlap("连接池", "") == 0.0


class TestMisc:
    def test_stable_id_is_deterministic(self):
        assert stable_id("a", "b") == stable_id("a", "b")
        assert stable_id("a", "b") != stable_id("a", "c")

    def test_normalize_collapses_spaces_but_keeps_paragraphs(self):
        assert normalize("a   b") == "a b"
        assert "\n\n" in normalize("a\n\n\n\nb")

    def test_content_tokens_prefers_bigrams(self):
        assert all(len(t) == 2 for t in content_tokens("连接池"))

    def test_snippet_truncates_on_sentence_boundary(self):
        text = "第一句话。第二句话。" * 20
        result = snippet(text, limit=30)
        assert result.endswith("。") or result.endswith("…")
