"""启发式重排测试：验证覆盖率精排能压过 RRF 粗排名次，并锁定 top_n 截断与标记。"""
from __future__ import annotations

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


class TestRerankCoverage:
    def test_prefers_higher_coverage_over_rrf_score(self):
        query = "连接池耗尽"
        cand_a = _hit("A", "数据库连接池耗尽需要降载", score=0.1)
        cand_b = _hit("B", "连接", score=0.9)

        result = rerank(query, [cand_b, cand_a], top_n=2)

        assert _ids(result) == ["A", "B"]
        assert result[0].score == 1.0
        assert all(s.retriever == "rerank" for s in result)

    def test_output_score_is_coverage(self):
        query = "熔断错误率阈值"
        candidates = [_hit("c1", "熔断错误率阈值为 50%", score=0.02)]

        result = rerank(query, candidates, top_n=1)

        assert 0.0 <= result[0].score <= 1.0
        assert result[0].score != 0.02


class TestRerankBoundaries:
    def test_truncates_to_top_n(self):
        query = "连接池"
        candidates = [
            _hit("a", "连接池配置", score=0.5),
            _hit("b", "连接池监控", score=0.4),
            _hit("c", "连接池告警", score=0.3),
            _hit("d", "数据库主从", score=0.2),
            _hit("e", "值班交接", score=0.1),
        ]

        result = rerank(query, candidates, top_n=2)

        assert len(result) == 2

    def test_empty_candidates(self):
        assert rerank("任意查询", [], top_n=5) == []