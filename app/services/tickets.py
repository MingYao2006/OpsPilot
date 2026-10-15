"""工单持久化：把 Ticket 落到一个 JSON 数组文件（照 KnowledgeBase manifest 的读写风格）。

编号规则：取现有工单里最大的数字后缀 +1，而不是 len(rows)+1。
后者在「删除过中间工单」或「并发建单」时会重新发出已被占用的编号
（例如删掉 TCK-0001 后，len=1 → 再发 TCK-0002，与已存在的单撞号）。

并发安全分两层：
- threading.Lock 挡住同进程多线程
- FileLock 挡住多进程（uvicorn 多 worker / 同时跑测试脚本）
- 写入走「临时文件 + os.replace」原子替换，避免读到写了一半的文件

已知边界：这是单机文件存储，多实例部署需换数据库。README 中已如实标注。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading

from app.config import Settings
from app.models import Ticket, TicketIn
from app.utils.filelock import FileLock

_ID_NO = re.compile(r"^TCK-(\d+)$")


class TicketStore:
    def __init__(self, settings: Settings) -> None:
        self.path = settings.ticket_store
        self._lock = threading.Lock()

    # ---------- 写入 ----------

    @staticmethod
    def _next_no(rows: list[dict]) -> int:
        """最大数字后缀 +1；无法解析的编号忽略，空表从 1 开始。"""
        highest = 0
        for row in rows:
            match = _ID_NO.match(str(row.get("ticket_id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def create(self, payload: TicketIn) -> Ticket:
        # 读-改-写整体串行：线程锁防同进程并发，文件锁防跨进程并发
        with self._lock, FileLock(self.path):
            rows = self._load()
            ticket_id = f"TCK-{self._next_no(rows):04d}"
            new_ticket = Ticket(ticket_id=ticket_id, status="open", **payload.model_dump())
            rows.append(new_ticket.model_dump(mode="json"))
            self._save(rows)
            return new_ticket

    def close(self, ticket_id: str) -> bool:
        with self._lock, FileLock(self.path):
            rows = self._load()
            for row in rows:
                if row.get("ticket_id") == ticket_id:
                    row["status"] = "closed"
                    self._save(rows)
                    return True
            return False

    # ---------- 读取 ----------

    def list(self) -> list[Ticket]:
        return [Ticket(**row) for row in self._load()]

    def get(self, ticket_id: str) -> Ticket | None:
        """按 id 取单条；找不到返回 None（纯数据层，不抛异常，404 判定留给路由）。"""
        for row in self._load():
            if row.get("ticket_id") == ticket_id:
                return Ticket(**row)
        return None

    def count(self) -> int:
        return len(self._load())

    # ---------- 内部（照 KnowledgeBase 的 _load/_save_manifest） ----------

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, rows: list[dict]) -> None:
        """原子写入：先写同目录临时文件，再 os.replace 覆盖。

        直接 write_text 会先截断再写，另一个进程此刻读到的是半个 JSON
        （_load 会静默返回 []，表现为工单凭空消失）。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(rows, ensure_ascii=False, indent=1)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, self.path)   # 同分区内原子
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise