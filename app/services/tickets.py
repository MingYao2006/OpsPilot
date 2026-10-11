"""工单持久化：把 Ticket 落到一个 JSON 数组文件（照 KnowledgeBase manifest 的读写风格）。"""
from __future__ import annotations

import json

from app.config import Settings
from app.models import Ticket, TicketIn


class TicketStore:
    def __init__(self, settings: Settings) -> None:
        self.path = settings.ticket_store

    # ---------- 写入 ----------

    def create(self, payload: TicketIn) -> Ticket:
        rows = self._load()
        next_no = len(rows) + 1
        ticket_id = f"TCK-{next_no:04d}"
        new_ticket = Ticket(ticket_id=ticket_id, status="open", **payload.model_dump())
        rows.append(new_ticket.model_dump(mode="json"))
        self._save(rows)
        return new_ticket

    def close(self, ticket_id: str) -> bool:
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