"""文档解析：Markdown / 文本文件的 frontmatter + 正文；PDF 为可选支持。

示例知识库使用轻量 frontmatter 声明元数据：

    ---
    title: 数据库连接池耗尽处置手册
    service: pay-gateway
    tags: [数据库, 连接池, 故障]
    ---
    正文……
"""

from __future__ import annotations

from pathlib import Path

from app.utils.logging import get_logger
from app.utils.text import normalize

log = get_logger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".log", ".rst", ".yaml", ".yml"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析可选的 frontmatter，返回 (元数据, 正文)。"""
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---"):
        return {}, normalized

    end = normalized.find("\n---", 3)
    if end == -1:
        return {}, normalized

    block = normalized[3:end]
    body = normalized[end + 4 :].lstrip("\n")

    meta: dict = {}
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        else:
            meta[key] = value.strip("'\"")
    return meta, body


def infer_title(meta: dict, body: str, fallback: str) -> str:
    if meta.get("title"):
        return str(meta["title"])
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
        elif stripped:
            break
    return fallback


def read_text_file(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_frontmatter(text)


def read_pdf_bytes(data: bytes) -> str:
    """PDF 为可选能力：未安装 pypdf 时给出明确指引而不是静默失败。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要 pypdf，请执行：pip install pypdf") from exc

    import io

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def read_bytes(filename: str, data: bytes) -> tuple[dict, str]:
    """从上传内容读取 (元数据, 正文)。"""
    suffix = Path(filename).suffix.lower()

    if suffix in PDF_SUFFIXES:
        return {}, read_pdf_bytes(data)

    if suffix in TEXT_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
        return parse_frontmatter(text)

    raise ValueError(
        f"暂不支持的文件类型 {suffix or '(无扩展名)'}；"
        f"当前支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def iter_knowledge_files(directory: Path) -> list[Path]:
    """递归列出知识库目录下所有受支持的文件。"""
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def service_from_meta(meta: dict) -> str | None:
    value = meta.get("service")
    return str(value) if value else None


def tags_from_meta(meta: dict) -> list[str]:
    value = meta.get("tags")
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def body_preview(body: str, limit: int = 200) -> str:
    return normalize(body)[:limit]
