"""BM25 关键词检索器：与向量召回互补，靠稀有词高 IDF 补足术语检索。"""

from __future__ import annotations

import math
from collections import Counter

from app.models import Chunk, ScoredChunk
from app.services.vector_store.base import VectorStore
from app.utils.text import tokenize


class BM25Index:
    def __init__(self, store: VectorStore, *, k1: float = 1.5, b: float = 0.75) -> None:
        # 外部依赖：数据源与算法参数
        self._store = store
        self._k1 = k1
        self._b = b

        # 内部状态：索引表，构造时全部为空，首次 search 时由 _build_index 填充
        self._docs: list[Chunk] = []
        self._ids: dict[str, int] = {}  # chunk_id → _docs 下标
        self._tf: list[Counter] = []  # 每块词频（与 _docs 下标对齐）
        self._dl: list[int] = []  # 每块 token 数
        self._df: dict[str, int] = {}  # token → 出现过的块数
        self._n = 0
        self._avgdl = 0.0
        self._built = False  # 是否已建索引（与 _fingerprint 联合判断）

    def search(
            self, query: str, *, top_k: int = 8, service: str | None = None
    ) -> list[ScoredChunk]:
        # 惰性构建 + 指纹失效：count 是廉价指纹。
        # 已知盲区：删除与新增数量持平时指纹不变、索引陈旧；
        # 实践中 replace 入库几乎必然改变块数，接受此局限并注释说明。
        if not self._built or self._store.count() != self._n:
            self._build_index()
    
        # 空库或全停用词时 avgdl==0，_score 会除零；在入口一次性挡住，
        # 比在每块打分循环里重复判断更省（循环外防御原则）
        if not self._n or not self._avgdl:
            return []
    
        tokens = tokenize(query)
        if not tokens:
            return []
    
        scored: list[ScoredChunk] = []
        for index, chunk in enumerate(self._docs):
            # service 过滤在打分层做：换 service 查询不必重建索引
            if service and chunk.service != service:
                continue
            score = self._score(tokens, index)
            # 零分不收：BM25 的 0 分是「零词命中」的纯噪音（不同于余弦的 0），
            # 收进来会在 WP2 的 RRF 里白占候选名次
            if score <= 0:
                continue
            scored.append(ScoredChunk(chunk=chunk, score=score, retriever="bm25"))
    
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def _score(self, query_tokens: list[str], index: int) -> float:
        """对 _docs[index] 这一个块计算 BM25 分数。"""

        score = 0.0
        tf = self._tf[index]
        dl = self._dl[index]

        for token in query_tokens:
            if token in tf:
                df = self._df[token]
                idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
                norm = self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                tf_part = tf[token] * (self._k1 + 1) / (tf[token] + norm)
                score += idf * tf_part
        return score



    def _build_index(self) -> None:
        """从向量库全量块构建词频/长度/df 表。"""
        chunks = self._store.all_chunks()

        docs: list[Chunk] = []
        ids: dict[str, int] = {}
        tf: list[Counter] = []
        dl: list[int] = []
        df: dict[str, int] = {}

        for chunk in chunks:
            counter = Counter(tokenize(chunk.text))
            # chunk_id 重复时保留先出现的，避免下标歧义
            if chunk.chunk_id in ids:
                continue
            ids[chunk.chunk_id] = len(docs)
            docs.append(chunk)
            tf.append(counter)
            dl.append(sum(counter.values()))
            # df 数的是「块数」：对块的词集遍历，同块内同词天然只 +1
            for token in counter:
                df[token] = df.get(token, 0) + 1

        self._docs = docs
        self._ids = ids
        self._tf = tf
        self._dl = dl
        self._df = df
        self._n = len(docs)
        self._avgdl = (sum(dl) / len(dl)) if dl else 0.0
        self._built = True
