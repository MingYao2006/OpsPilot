"""工单持久化：把 Ticket 落到一个 JSON 数组文件（照 KnowledgeBase manifest 的读写风格）。

编号规则：取现有工单里最大的数字后缀 +1，而不是 len(rows)+1。
后者在「删除过中间工单」或「并发建单」时会重新发出已被占用的编号
（例如删掉 TCK-0001 后，len=1 → 再发 TCK-0002，与已存在的单撞号）。
进程内并发由 _lock 串行化；跨进程仍以文件为准，属已知边界。
"""
from __future__ import annotations

import json
import re
import threading

from app.config import Settings
from app.models import Ticket, TicketIn

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
        # 读-改-写必须整体串行，否则两个请求会读到同一份 rows 并发出同一个编号
        with self._lock:
            rows = self._load()
            ticket_id = f"TCK-{self._next_no(rows):04d}"
            new_ticket = Ticket(ticket_id=ticket_id, status="open", **payload.model_dump())
            rows.append(new_ticket.model_dump(mode="json"))
            self._save(rows)
            return new_ticket

    def close(self, ticket_id: str) -> bool:
        with self._lock:
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )