"""文档解析测试：frontmatter、标题推断、编码与不支持类型的处理。"""

from __future__ import annotations

import pytest

from app.services.document_loader import (
    infer_title,
    iter_knowledge_files,
    parse_frontmatter,
    read_bytes,
    service_from_meta,
    tags_from_meta,
)


class TestParseFrontmatter:
    def test_parses_scalar_fields(self):
        meta, body = parse_frontmatter("---\ntitle: 手册\nservice: pay\n---\n正文\n")
        assert meta["title"] == "手册"
        assert meta["service"] == "pay"
        assert body.strip() == "正文"

    def test_parses_inline_list(self):
        meta, _ = parse_frontmatter("---\ntags: [数据库, 连接池]\n---\n正文")
        assert meta["tags"] == ["数据库", "连接池"]

    def test_strips_quotes(self):
        meta, _ = parse_frontmatter('---\ntitle: "带引号"\n---\n正文')
        assert meta["title"] == "带引号"

    def test_no_frontmatter_returns_body_unchanged(self):
        meta, body = parse_frontmatter("# 标题\n正文")
        assert meta == {}
        assert body == "# 标题\n正文"

    def test_unterminated_frontmatter_is_treated_as_body(self):
        meta, body = parse_frontmatter("---\ntitle: x\n没有结束标记")
        assert meta == {}
        assert "title: x" in body


class TestInferTitle:
    def test_meta_title_wins(self):
        assert infer_title({"title": "元数据标题"}, "# 正文标题", "兜底") == "元数据标题"

    def test_falls_back_to_first_heading(self):
        assert infer_title({}, "\n# 正文标题\n内容", "兜底") == "正文标题"

    def test_falls_back_to_default_when_no_heading(self):
        assert infer_title({}, "正文没有标题", "兜底") == "兜底"

    def test_ignores_leading_blank_lines(self):
        assert infer_title({}, "\n\n## 二级标题\n正文", "兜底") == "二级标题"


class TestReadBytes:
    def test_reads_plain_text(self):
        meta, body = read_bytes("note.txt", "# 标题\n内容".encode("utf-8"))
        assert "内容" in body

    def test_reads_utf8_chinese_correctly(self):
        _, body = read_bytes("note.md", "连接池耗尽处置".encode("utf-8"))
        assert "连接池耗尽处置" in body

    def test_unsupported_suffix_raises(self):
        with pytest.raises(ValueError, match="暂不支持"):
            read_bytes("data.xlsx", b"whatever")

    def test_malformed_utf8_does_not_crash(self):
        _, body = read_bytes("note.txt", b"\xff\xfe invalid bytes")
        assert isinstance(body, str)


class TestMetaHelpers:
    def test_service_from_meta(self):
        assert service_from_meta({"service": "pay"}) == "pay"
        assert service_from_meta({}) is None

    def test_tags_accepts_list_and_string(self):
        assert tags_from_meta({"tags": ["a", "b"]}) == ["a", "b"]
        assert tags_from_meta({"tags": "a, b"}) == ["a", "b"]
        assert tags_from_meta({}) == []


class TestIterKnowledgeFiles:
    def test_finds_supported_files_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.md").write_text("内容", encoding="utf-8")
        (tmp_path / "sub" / "b.txt").write_text("内容", encoding="utf-8")
        (tmp_path / "c.bin").write_bytes(b"\x00\x01")

        found = {p.name for p in iter_knowledge_files(tmp_path)}
        assert found == {"a.md", "b.txt"}

    def test_missing_directory_returns_empty(self, tmp_path):
        assert iter_knowledge_files(tmp_path / "nope") == []
