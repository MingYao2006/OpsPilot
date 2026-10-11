"""知识库编排：解析 → 分块 → 批量向量化 → 落库。

两个工程要点：
1. 批量 embedding —— 每批 embedding_batch_size 条，避免整篇文档一次性驻留内存
   （做法参考 LibreChat-AI/rag-api 的 batch 处理与内存边界控制）
2. 入库清单 —— 记录文件哈希，重启时跳过内容未变且已在库中的文档，避免重复计费
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.config import Settings
from app.models import DocumentIn, DocumentSummary, IngestResponse, IngestResult
from app.services.chunking import chunk_document, document_id_of
from app.services.document_loader import (
    iter_knowledge_files,
    read_text_file,
    service_from_meta,
    tags_from_meta,
    infer_title,
)
from app.services.embeddings import Embedder
from app.services.vector_store.base import VectorStore
from app.utils.logging import get_logger

log = get_logger(__name__)


class KnowledgeBase:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.manifest_path = settings.state_dir / "ingest_manifest.json"
        self.last_notes: list[str] = []

    # ---------- 写入 ----------

    def ingest(self, documents: list[DocumentIn], replace: bool = True) -> IngestResponse:
        results: list[IngestResult] = []
        total = 0

        for doc in documents:
            doc_id = document_id_of(doc)
            replaced = False
            if replace:
                removed = self.store.delete_document(doc_id)
                replaced = removed > 0

            chunks = chunk_document(
                doc,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            written = self._embed_and_upsert(chunks)
            total += written
            results.append(
                IngestResult(doc_id=doc_id, title=doc.title, chunks=written, replaced=replaced)
            )

        return IngestResponse(ingested=results, total_chunks=total)

    def _embed_and_upsert(self, chunks) -> int:
        """按批向量化并写入，控制单次内存占用。"""
        if not chunks:
            return 0

        written = 0
        batch_size = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(chunks), batch_size):
            window = chunks[start : start + batch_size]
            vectors = self.embedder.embed_documents([c.text for c in window])
            written += self.store.upsert(window, vectors)
        return written

    # ---------- 文件与目录 ----------

    def ingest_path(self, path: Path, *, root: Path | None = None) -> IngestResult:
        meta, body = read_text_file(path)
        base = root or path.parent
        try:
            relative = path.relative_to(base)
        except ValueError:
            relative = path.name

        doc = DocumentIn(
            title=infer_title(meta, body, fallback=path.stem),
            text=body,
            source=str(relative).replace("\\", "/"),
            service=service_from_meta(meta),
            tags=tags_from_meta(meta),
        )
        return self.ingest([doc], replace=True).ingested[0]

    def load_directory(self, directory: Path | None = None, *, force: bool = False) -> dict:
        """把一个目录下的知识库文件全部入库；内容未变且已在库中则跳过。"""
        root = directory or self.settings.knowledge_dir
        files = iter_knowledge_files(root)
        if not files:
            self.last_notes.append(f"知识库目录为空或不存在：{root}")
            return {"files": 0, "ingested": 0, "skipped": 0, "chunks": 0, "directory": str(root)}

        manifest = self._load_manifest()
        existing = {d["doc_id"]: d["chunks"] for d in self.store.list_documents()}

        ingested = skipped = total_chunks = 0
        for path in files:
            digest = self._file_digest(path)
            meta, body = read_text_file(path)
            doc = DocumentIn(
                title=infer_title(meta, body, fallback=path.stem),
                text=body,
                source=str(path.relative_to(root)).replace("\\", "/"),
                service=service_from_meta(meta),
                tags=tags_from_meta(meta),
            )
            doc_id = document_id_of(doc)
            key = str(path.relative_to(root)).replace("\\", "/")

            if not force and manifest.get(key, {}).get("sha1") == digest and existing.get(doc_id):
                skipped += 1
                continue

            result = self.ingest([doc], replace=True).ingested[0]
            manifest[key] = {"sha1": digest, "doc_id": doc_id, "chunks": result.chunks}
            ingested += 1
            total_chunks += result.chunks

        self._save_manifest(manifest)
        summary = {
            "files": len(files),
            "ingested": ingested,
            "skipped": skipped,
            "chunks": total_chunks,
            "directory": str(root),
        }
        log.info("知识库加载完成：%s", summary)
        return summary

    # ---------- 读取与管理 ----------

    def list_documents(self) -> list[DocumentSummary]:
        return [DocumentSummary(**row) for row in self.store.list_documents()]

    def delete_document(self, doc_id: str) -> int:
        return self.store.delete_document(doc_id)

    def count(self) -> int:
        return self.store.count()

    def set_service_scope(self, service: str | None) -> str | None:
        """检索前的权限/范围收口点。

        M1 只做服务过滤；M4 会在这里接入调用方身份，
        确保越权内容在排序之前就被排除（做法参考 LibreChat-AI/rag-api 的 scope.py）。
        """
        return service

    def stats(self) -> dict:
        return {
            "chunks": self.count(),
            "documents": len(self.store.list_documents()),
            "embedder": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "store": self.store.name,
        }

    # ---------- 内部 ----------

    @staticmethod
    def _file_digest(path: Path) -> str:
        return hashlib.sha1(path.read_bytes()).hexdigest()

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )
