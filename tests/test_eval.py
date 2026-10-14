"""评测脚本的判据测试。

重点锁死一个曾经的 bug：阴性对照用 min_relevance(0.06) 去卡混合检索的 RRF 分，
而两路 RRF 的理论上限只有 2/(60+1)≈0.0328 —— `all(score < 0.06)` 恒为真，
阴性对照永远"通过"，这个刹车形同虚设。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval as eval_mod  # noqa: E402
from app.models import Chunk, ScoredChunk  # noqa: E402


def _hit(text: str, source: str = "01-支付网关.md", score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id="c1", doc_id="d1", text=text, source=source),
        score=score,
        retriever="rrf",
    )


class TestHitJudgement:
    def test_exact_prefix(self):
        assert eval_mod.hit("01-支付网关.md", "01")

    def test_or_across_documents(self):
        assert eval_mod.hit("06-复盘.md", "01|06")
        assert not eval_mod.hit("07-慢查询.md", "01|06")

    def test_none_never_hits(self):
        assert not eval_mod.hit("01-支付网关.md", "NONE")


class TestNegativeControl:
    """阴性判据必须量纲无关，且真的能被触发。"""

    def test_no_hits_passes(self):
        assert eval_mod.is_negative_correct([], "红烧肉怎么做")

    def test_unrelated_query_passes(self):
        hits = [_hit("连接池耗尽需要先降载，再定位根因。")]
        assert eval_mod.is_negative_correct(hits, "红烧肉怎么做才好吃")

    def test_topically_matching_query_fails(self):
        """阴性用例若真被知识库内容覆盖，必须判不通过 —— 否则判据是恒真的。"""
        hits = [_hit("支付网关连接池耗尽时先降载再定位根因。")]
        assert not eval_mod.is_negative_correct(hits, "支付网关连接池耗尽怎么处理")

    def test_old_score_threshold_would_be_vacuous(self):
        """回归锁：用 min_relevance 卡 RRF 分，永远判通过（这就是那个 bug）。"""
        settings = eval_mod.get_settings()
        ceiling = eval_mod.rrf_ceiling(settings)
        assert ceiling < settings.min_relevance, "前提变了，请重新审视该回归测试"

        # 即使分数取到理论上限，旧判据依然"通过"
        best_possible = [_hit("完全无关的内容", score=ceiling)]
        old_style = all(h.score < settings.min_relevance for h in best_possible)
        assert old_style is True, "旧判据在上限分数下也应恒为真（证明它没有区分力）"

    def test_overlap_metric_discriminates(self):
        """新判据：同一对输入下，相关/不相关能被分开。"""
        relevant = _hit("支付网关连接池耗尽时先降载，再定位根因。")
        irrelevant = _hit("红烧肉的烹饪步骤与火候控制。")
        assert not eval_mod.is_negative_correct([relevant], "支付网关连接池耗尽怎么处理")
        assert eval_mod.is_negative_correct([irrelevant], "支付网关连接池耗尽怎么处理")


class TestSanityCheck:
    def test_threshold_is_inside_unit_interval(self):
        assert 0 < eval_mod.NEGATIVE_OVERLAP_CEILING < 1

    def test_no_problems_with_default_settings(self):
        assert eval_mod.sanity_check(eval_mod.get_settings()) == []


class TestReciprocalRank:
    def test_first_position(self):
        hits = [_hit("内容", source="05-熔断.md")]
        assert eval_mod.reciprocal_rank(hits, "05") == 1.0

    def test_second_position(self):
        hits = [_hit("内容", source="07-x.md"), _hit("内容", source="05-熔断.md")]
        assert eval_mod.reciprocal_rank(hits, "05") == 0.5

    def test_no_match_is_zero(self):
        hits = [_hit("内容", source="07-x.md")]
        assert eval_mod.reciprocal_rank(hits, "05") == 0.0


class TestCasesIntegrity:
    def test_every_category_is_covered(self):
        cats = {c for _, _, c in eval_mod.CASES}
        assert cats == {"正面", "换词", "跨文档", "阴性"}

    def test_expectations_are_well_formed(self):
        for query, expect, cat in eval_mod.CASES:
            assert query.strip(), f"空查询：{cat}"
            if expect == "NONE":
                assert cat == "阴性", f"{query} 用了 NONE 但类别是 {cat}"
            else:
                assert cat != "阴性", f"{query} 是阴性类别却没标 NONE"
                for part in expect.split("|"):
                    assert len(part) == 2 and part.isdigit(), f"非法期望：{expect}"


@pytest.mark.parametrize("use_rerank", [False, True])
def test_separability_gap_is_real(container, use_rerank):
    """在两种模式下，阳性最低覆盖率都要高于阴性最高覆盖率，且阈值落在空隙内。"""
    pos_min, neg_max = eval_mod.separability_report(container, use_rerank)
    assert neg_max < eval_mod.NEGATIVE_OVERLAP_CEILING < pos_min, (
        f"use_rerank={use_rerank}: 空隙 ({neg_max:.3f}, {pos_min:.3f}) "
        f"未包含阈值 {eval_mod.NEGATIVE_OVERLAP_CEILING}"
    )
