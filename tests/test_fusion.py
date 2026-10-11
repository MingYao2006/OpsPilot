"""RRF 倒数排名融合测试：锁定「只按名次共识排序」的核心行为。"""

from __future__ import annotations

from app.models import Chunk, ScoredChunk
from app.services.retrieval.fusion import fuse


def _hit(cid: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, doc_id="d1", text=f"块-{cid}"),
        score=score,
        retriever="vector",
    )


def _ids(scored: list[ScoredChunk]) -> list[str]:
    """把结果里的 chunk_id 按顺序抽出来，方便断言排序。"""
    return [s.chunk.chunk_id for s in scored]


class TestRankConsensus:
    def test_multi_list_hit_ranks_above_single(self):
        # 路1：c1 第1名, c2 第2名
        ranking_vec = [_hit("c1"), _hit("c2")]
        # 路2：c3 第1名, c1 第2名   → c1 在「两路」都被认可
        ranking_bm25 = [_hit("c3"), _hit("c1")]

        result = fuse([ranking_vec, ranking_bm25], top_k=3)

        assert _ids(result) == ["c1", "c3", "c2"]


class TestRankOnlyNotScore:
    def test_ignores_raw_score_magnitude(self):
        # 名次在前(cx)但原始分故意设低，名次在后(cy)但原始分故意设高
        ranking = [_hit("cx", score=0.01), _hit("cy", score=0.99)]

        result = fuse([ranking], top_k=2)
        assert _ids(result) == ["cx", "cy"]


class TestDedup:
    def test_same_chunk_appears_once(self):
        # 同一个 c1 在两路都排第1
        result = fuse([[_hit("c1")], [_hit("c1")]], top_k=5)
        assert len(result) == 1

class TestRetrieverLabel:
    def test_all_output_labeled_rrf(self):
        result = fuse([[_hit("a"), _hit("b")]], top_k=5)
        assert all(s.retriever == "rrf" for s in result)

class TestBoundaries:
    def test_truncates_to_top_k(self):
        result = fuse([[_hit("a"), _hit("b"), _hit("c")]], top_k=2)
        assert len(result) == 2
    def test_empty_input(self):
        assert fuse([], top_k=3) == []

class TestWeighting:
    def test_weights_can_flip_order(self):
        rankings = [[_hit("cx")], [_hit("cy")]]
        # 第1路权重高 → 谁第一?填期望的 _ids(...)
        assert _ids(fuse(rankings, top_k=2, weights=[1.0, 2.0])) == ["cy","cx"]
        # 第0路权重高 → 谁第一?填期望的 _ids(...)
        assert _ids(fuse(rankings, top_k=2, weights=[2.0, 1.0])) == ["cx","cy"]

    def test_none_weights_equals_all_one(self):
        rankings = [[_hit("cx")], [_hit("cy")]]
        assert _ids(fuse(rankings, top_k=2)) == _ids(fuse(rankings, top_k=2, weights=[1.0, 1.0]))
