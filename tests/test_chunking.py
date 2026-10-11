"""分块测试：块长约束、重叠、ID 稳定性、超长句处理。"""

from __future__ import annotations

import pytest

from app.models import DocumentIn
from app.services.chunking import chunk_document, document_id_of, pack_sentences


def _sentences(count: int) -> list[str]:
    return [f"句子{i:02d}。" for i in range(count)]


class TestPackSentences:
    def test_respects_size_limit(self):
        blocks = pack_sentences(_sentences(12), size=12, overlap=5)
        assert blocks, "不应产生空结果"
        assert all(len(b) <= 12 for b in blocks)

    def test_produces_multiple_blocks(self):
        blocks = pack_sentences(_sentences(12), size=12, overlap=5)
        assert len(blocks) > 1

    def test_consecutive_blocks_overlap(self):
        blocks = pack_sentences(_sentences(12), size=12, overlap=5)
        for previous, current in zip(blocks, blocks[1:]):
            assert previous.endswith(current[:5])

    def test_zero_overlap_has_no_duplication(self):
        blocks = pack_sentences(_sentences(12), size=10, overlap=0)
        joined = "".join(blocks)
        assert joined.count("句子00。") == 1

    def test_oversized_sentence_is_hard_split(self):
        blocks = pack_sentences(["甲" * 100], size=30, overlap=0)
        assert len(blocks) > 1
        assert all(len(b) <= 30 for b in blocks)

    def test_invalid_overlap_rejected(self):
        with pytest.raises(ValueError):
            pack_sentences(_sentences(3), size=10, overlap=10)

    def test_empty_input(self):
        assert pack_sentences([], size=10, overlap=2) == []


class TestChunkDocument:
    def _doc(self, **overrides) -> DocumentIn:
        payload = {
            "title": "连接池处置手册",
            "text": "第一段内容。" * 20 + "\n\n" + "第二段内容。" * 20,
            "source": "kb/01.md",
            "service": "pay-gateway",
            "tags": ["数据库"],
        }
        payload.update(overrides)
        return DocumentIn(**payload)

    def test_metadata_propagates_to_every_chunk(self):
        chunks = chunk_document(self._doc(), chunk_size=120, chunk_overlap=20)
        assert chunks
        for chunk in chunks:
            assert chunk.title == "连接池处置手册"
            assert chunk.source == "kb/01.md"
            assert chunk.service == "pay-gateway"
            assert chunk.tags == ["数据库"]

    def test_ordinals_are_sequential(self):
        chunks = chunk_document(self._doc(), chunk_size=120, chunk_overlap=20)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_chunk_ids_are_stable_across_runs(self):
        first = chunk_document(self._doc(), chunk_size=120, chunk_overlap=20)
        second = chunk_document(self._doc(), chunk_size=120, chunk_overlap=20)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_different_content_yields_different_ids(self):
        a = chunk_document(self._doc(), chunk_size=120, chunk_overlap=20)
        b = chunk_document(self._doc(text="完全不同的内容。" * 30), chunk_size=120, chunk_overlap=20)
        assert {c.chunk_id for c in a} != {c.chunk_id for c in b}

    def test_respects_chunk_size(self):
        chunks = chunk_document(self._doc(), chunk_size=120, chunk_overlap=20)
        assert all(len(c.text) <= 120 for c in chunks)

    def test_explicit_doc_id_is_used(self):
        doc = self._doc(doc_id="my-doc")
        assert document_id_of(doc) == "my-doc"
        assert all(c.doc_id == "my-doc" for c in chunk_document(doc, chunk_size=120, chunk_overlap=20))

    def test_generated_doc_id_is_deterministic(self):
        assert document_id_of(self._doc()) == document_id_of(self._doc())
