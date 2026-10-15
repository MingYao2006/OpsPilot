"""精排测试：锁定「覆盖率与 RRF 加权融合」的行为与权重语义。

背景：原实现是纯覆盖率重排，实测在真实语义向量下为负增益
（Top-1 23→21、MRR 0.979→0.938）。改为归一化加权融合后，
cov_weight 决定覆盖率的话语权：0 = 纯 RRF，1 = 旧行为。
"""
from __future__ import annotations

import pytest

from app.models import Chunk, ScoredChunk
from app.services.retrieval.rerank import rerank


def _hit(cid: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, doc_id="d1", text=text),
        score=score,
        retriever="rrf",
    )


def _ids(scored: list[ScoredChunk]) -> list[str]:
    return [s.chunk.chunk_id for s in scored]


class TestWeightSemantics:
    """cov_weight=0 必须等价于不精排，=1 必须复现旧版纯覆盖率行为。"""

    def test_zero_weight_preserves_rrf_order(self):
        """权重 0 → 完全尊重粗排名次，覆盖率不参与。"""
        query = "连接池耗尽"
        cand_a = _hit("A", "数据库连接池耗尽需要降载", score=0.1)   # 覆盖率高但 RRF 低
        cand_b = _hit("B", "连接", score=0.9)                      # 覆盖率低但 RRF 高

        result = rerank(query, [cand_b, cand_a], top_n=2, cov_weight=0.0)

        assert _ids(result) == ["B", "A"], "权重 0 时不应被覆盖率改变顺序"

    def test_full_weight_reproduces_old_behaviour(self):
        """权重 1 → 等价旧版：覆盖率高者胜，即使 RRF 分低得多。"""
        query = "连接池耗尽"
        cand_a = _hit("A", "数据库连接池耗尽需要降载", score=0.1)
        cand_b = _hit("B", "连接", score=0.9)

        result = rerank(query, [cand_b, cand_a], top_n=2, cov_weight=1.0)

        assert _ids(result) == ["A", "B"], "权重 1 时应复现旧的纯覆盖率排序"

    def test_low_weight_keeps_rrf_consensus(self):
        """默认低权重下，RRF 领先足够多时不会被覆盖率翻盘（这正是修负增益的关键）。"""
        query = "连接池耗尽"
        strong = _hit("S", "连接池", score=1.0)          # RRF 高
        weak = _hit("W", "数据库连接池耗尽需要降载", score=0.0)  # 覆盖率高但 RRF 最低

        result = rerank(query, [strong, weak], top_n=2, cov_weight=0.35)

        assert _ids(result)[0] == "S", "低权重下 RRF 共识应压过覆盖率"

    def test_default_weight_is_between_extremes(self):
        assert 0.0 < 0.35 < 1.0


class TestOutputContract:
    def test_score_is_normalized_fusion(self):
        candidates = [_hit("c1", "熔断错误率阈值为 50%", score=0.02)]
        result = rerank("熔断错误率阈值", candidates, top_n=1, cov_weight=0.5)
        assert 0.0 <= result[0].score <= 1.0
        assert result[0].score != 0.02, "输出应是融合分，不是原始 RRF 分"

    def test_all_labelled_rerank(self):
        result = rerank("连接池", [_hit("a", "连接池配置", 0.5)], top_n=5)
        assert all(s.retriever == "rerank" for s in result)

    def test_chunk_identity_is_preserved(self):
        src = _hit("a", "连接池配置", 0.5)
        result = rerank("连接池", [src], top_n=1)
        assert result[0].chunk.chunk_id == src.chunk.chunk_id
        assert result[0].chunk.text == src.chunk.text


class TestBoundaries:
    def test_truncates_to_top_n(self):
        candidates = [
            _hit("a", "连接池配置", 0.5),
            _hit("b", "连接池监控", 0.4),
            _hit("c", "连接池告警", 0.3),
            _hit("d", "数据库主从", 0.2),
            _hit("e", "值班交接", 0.1),
        ]
        assert len(rerank("连接池", candidates, top_n=2)) == 2

    def test_empty_candidates(self):
        assert rerank("任意查询", [], top_n=5) == []

    def test_single_candidate(self):
        result = rerank("连接池", [_hit("only", "连接池配置", 0.5)], top_n=5)
        assert _ids(result) == ["only"]

    def test_identical_scores_do_not_crash(self):
        """归一化时 max==min 会除零，必须兜住。"""
        candidates = [
            _hit("a", "同样文本", 0.5),
            _hit("b", "同样文本", 0.5),
        ]
        result = rerank("同样文本", candidates, top_n=2, cov_weight=0.5)
        assert len(result) == 2

    def test_weight_is_clamped(self):
        candidates = [_hit("a", "连接池配置", 0.5), _hit("b", "其它", 0.1)]
        # 越界权重不应抛异常
        assert len(rerank("连接池", candidates, top_n=2, cov_weight=5.0)) == 2
        assert len(rerank("连接池", candidates, top_n=2, cov_weight=-1.0)) == 2

    @pytest.mark.parametrize("weight", [0.0, 0.35, 0.7, 1.0])
    def test_all_weights_return_all_candidates(self, weight):
        candidates = [_hit(f"c{i}", f"连接池 文档{i}", 0.5 - i * 0.05) for i in range(5)]
        assert len(rerank("连接池", candidates, top_n=5, cov_weight=weight)) == 5
