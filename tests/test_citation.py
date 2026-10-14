"""引用构建与校验原语测试。重点是数字一致性 —— 运维场景下编造阈值代价极高。"""

from __future__ import annotations

import pytest

from app.models import Chunk, ScoredChunk
from app.services.citation import (
    build_citations,
    citation_support,
    sentence_support_report,
    strip_citation_markers,
    unsupported_numbers,
    used_citation_indexes,
    verify_answer,
)


def _hit(chunk_id: str, text: str, score: float = 0.5, source: str = "kb/01.md") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=text, title="连接池手册", source=source),
        score=score,
        retriever="vector",
    )


class TestBuildCitations:
    def test_indexes_start_at_one(self):
        citations = build_citations([_hit("c1", "甲"), _hit("c2", "乙")])
        assert [c.index for c in citations] == [1, 2]

    def test_carries_source_metadata(self):
        citation = build_citations([_hit("c1", "内容", source="kb/07.md")])[0]
        assert citation.doc_id == "d1"
        assert citation.title == "连接池手册"
        assert citation.source == "kb/07.md"
        assert citation.chunk_id == "c1"

    def test_score_is_rounded(self):
        citation = build_citations([_hit("c1", "内容", score=0.123456)])[0]
        assert citation.score == 0.1235

    def test_empty_input(self):
        assert build_citations([]) == []


class TestUnsupportedNumbers:
    def test_flags_fabricated_threshold(self):
        """答案里出现的阈值如果原文没有，必须被抓出来。"""
        fabricated = unsupported_numbers(
            "连接池上限为 200，超时 5 秒。",
            ["maximumPoolSize 为 50，connectionTimeout 为 3000ms。"],
        )
        assert "200" in fabricated
        assert "5" in fabricated

    def test_accepts_numbers_present_in_source(self):
        assert unsupported_numbers(
            "连接池上限为 50。",
            ["maximumPoolSize 为 50。"],
        ) == set()

    def test_percent_sign_is_tolerated(self):
        """50 与 50% 视为同一个数，避免格式差异造成误报。"""
        assert unsupported_numbers("错误率 50%", ["错误率阈值 50%"]) == set()

    def test_number_split_across_multiple_sources(self):
        assert unsupported_numbers(
            "阈值 50，超时 3000ms。",
            ["阈值 50。", "超时 3000ms。"],
        ) == set()

    def test_empty_source_flags_all_numbers(self):
        assert unsupported_numbers("阈值 50", []) == {"50"}

    def test_citation_markers_are_not_numbers(self):
        """[1] 是引用标记，不是答案编造的数字 —— 否则每条带引用的答案都会误报。"""
        assert unsupported_numbers("错误率阈值为 50% [1]。", ["错误率阈值为 50%。"]) == set()

    def test_multidigit_marker_ignored(self):
        assert unsupported_numbers("结论如下 [12]。", ["结论如下。"]) == set()

    def test_real_number_alongside_marker_still_flagged(self):
        assert unsupported_numbers("错误率阈值为 80% [1]。", ["错误率阈值为 50%。"]) == {"80", "80%"}


class TestStripCitationMarkers:
    def test_removes_markers(self):
        assert "[1]" not in strip_citation_markers("阈值 50% [1]，窗口 10 秒 [2]。")

    def test_keeps_other_brackets(self):
        assert "[附录A]" in strip_citation_markers("参见 [附录A]。")


class TestVerifyAnswer:
    _SRC = "熔断器在错误率超过 50% 时打开，滑动窗口为 10 秒。"

    def _cites(self, texts: list[str]):
        return build_citations(
            [ScoredChunk(chunk=Chunk(chunk_id=f"c{i}", doc_id="d", text=t), score=0.5)
             for i, t in enumerate(texts, start=1)]
        )

    def test_grounded_answer_passes(self):
        cites = self._cites([self._SRC])
        verdict = verify_answer("熔断器在错误率超过 50% 时打开 [1]。", cites, source_texts=[self._SRC])
        assert verdict.ok, verdict.reasons

    def test_fabricated_number_is_rejected(self):
        """运维场景最危险：把阈值编造成另一个数。"""
        cites = self._cites([self._SRC])
        verdict = verify_answer("熔断器在错误率超过 80% 时打开 [1]。", cites, source_texts=[self._SRC])
        assert verdict.refused
        assert "80" in verdict.fabricated_numbers
        assert any("数字" in r for r in verdict.reasons)

    def test_missing_citation_index_is_rejected(self):
        cites = self._cites([self._SRC])
        verdict = verify_answer("熔断器在错误率超过 50% 时打开 [7]。", cites, source_texts=[self._SRC])
        assert verdict.refused
        assert verdict.missing_citations == {7}

    def test_unsupported_sentences_are_rejected(self):
        cites = self._cites([self._SRC])
        answer = "公司年会安排在三楼。食堂今天供应红烧肉。熔断阈值见资料 [1]。"
        verdict = verify_answer(answer, cites, source_texts=[self._SRC])
        assert verdict.refused
        assert any("支撑" in r for r in verdict.reasons)

    def test_full_text_beats_truncated_snippet(self):
        """数字落在 snippet 160 字截断之后时，必须靠完整原文判为有据。"""
        long_src = "开头填充内容。" * 30 + "连接池上限为 50。"
        cites = self._cites([long_src])
        verdict = verify_answer("连接池上限为 50 [1]。", cites, source_texts=[long_src])
        assert verdict.ok, verdict.reasons

    def test_empty_answer_sentences_pass_vacuously(self):
        verdict = verify_answer("", self._cites([self._SRC]), source_texts=[self._SRC])
        assert verdict.ok


class TestCitationSupport:
    def test_supported_sentence_scores_high(self):
        score = citation_support("连接池耗尽需要先降载", ["连接池耗尽需要先降载，再定位。"])
        assert score > 0.9

    def test_unrelated_sentence_scores_low(self):
        score = citation_support("公司年会安排在三楼", ["连接池耗尽需要先降载。"])
        assert score < 0.2

    def test_takes_best_source(self):
        score = citation_support("熔断错误率阈值", ["完全无关内容", "熔断错误率阈值为 50%"])
        assert score > 0.9

    def test_no_sources(self):
        assert citation_support("任意内容", []) == 0.0


class TestUsedCitationIndexes:
    def test_extracts_markers(self):
        assert used_citation_indexes("根据 [1] 与 [3] 可知") == {1, 3}

    def test_ignores_non_numeric_brackets(self):
        assert used_citation_indexes("参见 [附录A] 与 [1]") == {1}

    def test_no_markers(self):
        assert used_citation_indexes("没有引用") == set()

    def test_sentence_report_length_matches(self):
        report = sentence_support_report("第一句。第二句。", ["第一句。"])
        assert len(report) == 2
        assert report[0][1] > report[1][1]
