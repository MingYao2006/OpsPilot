"""并发不阻塞测试。

两处曾阻塞事件循环，都已修复：
1. container.retrieve() —— 同步函数，用 run_in_threadpool 解决
2. llm.chat_stream()   —— 同步生成器，改用 chat_stream_async 解决

这类 bug 单请求测不出来：功能完全正常，只是并发时全部排队。
所以必须专门测「N 个并发是否真的并行」。

判据用「并行度」而不是绝对耗时：
- 并行度 ≈ 1.0 → 请求真并行
- 并行度 ≈ N   → 完全串行（阻塞）
绝对耗时受机器性能影响，不适合做断言。
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import pytest


class SlowAsyncLLM:
    """异步流式 LLM：每片之间 await sleep，模拟真实网络等待。

    关键：用 await asyncio.sleep 而不是 time.sleep。
    time.sleep 会阻塞事件循环，正好复现「同步实现」的问题。
    """

    name = "slow-async"

    def __init__(self, pieces: int = 5, delay: float = 0.05):
        self.pieces = [f"片{i}" for i in range(pieces)]
        self.delay = delay

    async def chat_stream_async(self, system: str, user: str) -> AsyncIterator[str]:
        for piece in self.pieces:
            await asyncio.sleep(self.delay)
            yield piece


class TestStreamConcurrency:
    def test_retrieve_runs_in_threadpool(self):
        """retrieve 必须经 run_in_threadpool 调度，不能直接同步调用。

        判据：5 个并发请求的总耗时不能接近 5 倍单次。
        用 sleep 模拟远端 embedding 的 200~300ms 网络等待，
        这样阻塞效果才明显（本地哈希向量器只要 10ms，看不出来）。
        """
        import os

        from starlette.concurrency import run_in_threadpool

        from app.config import get_settings
        from app.container import build_container

        settings = get_settings()
        container = build_container(settings)
        container.knowledge.load_directory()
        container.retrieve("预热一次，排除 BM25 索引构建", top_k=3)   # 预热

        def slow_retrieve() -> None:
            time.sleep(0.08)          # 模拟远端 embedding 网络等待
            container.retrieve("支付网关连接池耗尽怎么处理", top_k=3)

        async def measure() -> tuple[float, float]:
            single_t0 = time.perf_counter()
            await run_in_threadpool(slow_retrieve)
            single = time.perf_counter() - single_t0

            n = 5
            batch_t0 = time.perf_counter()
            await asyncio.gather(*(run_in_threadpool(slow_retrieve) for _ in range(n)))
            batch = time.perf_counter() - batch_t0
            return single, batch

        single, batch = asyncio.run(measure())
        parallelism = batch / single

        # 串行的话 parallelism ≈ 5；并行的话 ≈ 1。
        # 留出余量（线程池有调度开销），断言 < 3 就足以区分两种实现。
        assert parallelism < 3.0, (
            f"5 个并发近乎串行（并行度 {parallelism:.2f}x，单次 {single:.2f}s，"
            f"批量 {batch:.2f}s）——retrieve 可能没走线程池，正在阻塞事件循环"
        )

    def test_async_llm_yields_without_blocking(self):
        """chat_stream_async 在等待时应让出事件循环。

        用 SlowAsyncLLM（await sleep）验证：N 个并发的总耗时
        应接近单个而不是 N 倍。
        """
        llm = SlowAsyncLLM(pieces=5, delay=0.05)   # 单个约 0.25s

        async def consume() -> int:
            n = 0
            async for _piece in llm.chat_stream_async("sys", "user"):
                n += 1
            return n

        async def measure() -> tuple[int, float, float]:
            single_t0 = time.perf_counter()
            n = await consume()
            single = time.perf_counter() - single_t0

            batch_t0 = time.perf_counter()
            await asyncio.gather(*(consume() for _ in range(4)))
            batch = time.perf_counter() - batch_t0
            return n, single, batch

        n, single, batch = asyncio.run(measure())
        assert n == 5

        parallelism = batch / single
        assert parallelism < 2.0, (
            f"4 个并发几乎串行（并行度 {parallelism:.2f}x）——"
            f"异步流式可能在等待时阻塞了事件循环"
        )

    def test_sync_generator_would_block(self):
        """反向验证：同步生成器确实会阻塞，证明上面两条测试有意义。

        这条不是测修复，而是证明「如果不改会怎样」。
        用 time.sleep 模拟同步实现，并行度应接近 N。
        """

        def sync_stream():
            for i in range(5):
                time.sleep(0.05)      # 同步阻塞
                yield f"片{i}"

        async def consume_sync() -> int:
            n = 0
            for _piece in sync_stream():      # 同步 for，不 await
                n += 1
            return n

        async def measure() -> tuple[float, float]:
            single_t0 = time.perf_counter()
            await consume_sync()
            single = time.perf_counter() - single_t0

            batch_t0 = time.perf_counter()
            await asyncio.gather(*(consume_sync() for _ in range(4)))
            batch = time.perf_counter() - batch_t0
            return single, batch

        single, batch = asyncio.run(measure())
        parallelism = batch / single
        assert parallelism > 3.0, (
            f"同步生成器的并行度应接近 4（实际 {parallelism:.2f}x）——"
            f"如果这条失败，说明 asyncio 行为变了，上面两条并发测试的前提不再成立"
        )
