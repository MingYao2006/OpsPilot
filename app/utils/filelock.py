"""跨进程文件锁：Windows 用 msvcrt，POSIX 用 fcntl。

用于工单存储的读-改-写临界区。进程内线程锁（threading.Lock）只能挡住同一进程的
并发；uvicorn 多 worker 或同时跑测试与手工脚本时，两个进程会各自读到同一份文件并
写出冲突结果（典型表现：发出重复的工单编号，或后写的覆盖先写的）。
"""
from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType


class FileLock:
    """基于独立锁文件的排他锁，支持 with 语句。

    锁文件与被保护的数据文件分开（`<data>.lock`），这样数据文件可以自由地
    原子替换（os.replace），而锁始终落在同一个 inode/文件上。
    """

    def __init__(self, target: Path, timeout_s: float = 10.0) -> None:
        self.lock_path = target.with_suffix(target.suffix + ".lock")
        self.timeout_s = timeout_s
        self._fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # O_CREAT 让首次运行即可用；不删除锁文件，避免删除与加锁之间的竞态
        self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        self._acquire()
        return self

    def _acquire(self) -> None:
        assert self._fd is not None
        if os.name == "nt":
            import msvcrt

            # LK_LOCK 重试 10 次、每次间隔 1 秒后抛 OSError，故外层再包一层重试
            import time

            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"获取文件锁超时：{self.lock_path}") from None
                    time.sleep(0.05)
        else:
            import fcntl
            import time

            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"获取文件锁超时：{self.lock_path}") from None
                    time.sleep(0.05)

    def _release(self) -> None:
        if self._fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._release()
