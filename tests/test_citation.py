"""引用构建与校验原语测试。重点是数字一致性 —— 运维场景下编造阈值代价极高。"""

from __future__ import annotations

import pytest

from app.models import Chunk, ScoredChunk
from app.services.citation import (
    build_citations,
    citation_support,
    sentence_support_report,
    unsupported_numbers,
    used_citation_indexes,
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
