"""向量库统一接口：新增后端只需实现本抽象类并在 factory 中注册。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import Chunk, ScoredChunk


class VectorStore(ABC):
    """所有后端共用的最小契约。

    约定：`search` 返回的 score 必须落在 [0, 1] 且越大越相关，
    这样上层 min_relevance 门限才能跨后端复用。
    """

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """写入或覆盖；返回写入条数。"""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 8,
        service: str | None = None,
    ) -> list[ScoredChunk]:
        """按向量相似度检索。"""

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """删除某文档的全部块，返回删除条数。"""

    @abstractmethod
    def list_documents(self) -> list[dict]:
        """列出已入库文档的汇总信息。"""

    @abstractmethod
    def count(self) -> int: ...

    def all_chunks(self) -> list[Chunk]:
        """供 BM25 等需要全量的检索器使用。"""
        raise NotImplementedError

    def reset(self) -> None:
        """清空（测试与重建索引用）。"""
        raise NotImplementedError

    def health(self) -> dict:
        return {"backend": self.name, "dim": self.dim, "chunks": self.count()}
