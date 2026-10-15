"""工单持久化测试（全部落在 tmp_path，不碰 work/state 真数据）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.models import TicketIn
from app.services.tickets import TicketStore


@pytest.fixture
def store(tmp_path) -> TicketStore:
    settings = Settings(ticket_store=tmp_path / "tickets.json")
    return TicketStore(settings)


class TestCreate:
    def test_first_ticket_gets_sequential_id(self, store):
        ticket = store.create(TicketIn(title="支付网关超时", summary="错误率上升", priority="P1"))
        assert ticket.ticket_id == "TCK-0001"
        assert ticket.status == "open"
        assert ticket.priority == "P1"

    def test_ids_increment_with_existing(self, store):
        store.create(TicketIn(title="支付网关超时", summary="错误率上升", priority="P1"))
        ticket = store.create(TicketIn(title="支付网关超时", summary="错误率上升", priority="P1"))
        assert ticket.ticket_id == "TCK-0002"


class TestIdUniqueness:
    """编号唯一性回归：len(rows)+1 会在删单/并发时重发已占用的编号。"""

    def test_no_reuse_after_middle_row_removed(self, store):
        store.create(TicketIn(title="甲故障描述", summary="甲现象", priority="P1"))
        store.create(TicketIn(title="乙故障描述", summary="乙现象", priority="P2"))
        # 模拟运维手工删掉一张单（或并发场景下 rows 变短）
        rows = json.loads(store.path.read_text(encoding="utf-8"))
        rows.pop(0)
        store.path.write_text(json.dumps(rows), encoding="utf-8")

        fresh = store.create(TicketIn(title="丙故障描述", summary="丙现象", priority="P3"))
        ids = [t.ticket_id for t in store.list()]
        assert fresh.ticket_id == "TCK-0003", ids
        assert len(ids) == len(set(ids)), f"编号重复：{ids}"

    def test_concurrent_creates_get_unique_ids(self, store):
        import threading

        results: list[str] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(n: int) -> None:
            try:
                barrier.wait(timeout=10)  # 尽量让 8 个线程同时进入 create
                results.append(
                    store.create(TicketIn(title=f"并发故障{n}", summary="并发现象", priority="P3")).ticket_id
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, errors
        assert len(results) == 8
        assert len(set(results)) == 8, f"并发发出重复编号：{sorted(results)}"

    def test_unparseable_ids_do_not_break_counter(self, store):
        store.path.write_text(
            json.dumps([{"ticket_id": "legacy-abc"}, {"ticket_id": "TCK-0007"}]),
            encoding="utf-8",
        )
        fresh = store.create(TicketIn(title="新故障描述", summary="新现象", priority="P3"))
        assert fresh.ticket_id == "TCK-0008"


class TestCrossProcessSafety:
    """跨进程安全：threading.Lock 挡不住多进程，必须靠文件锁 + 原子写。

    用真实子进程验证（不是模拟）：每个子进程独立 import、独立打开文件，
    这才等价于 uvicorn 多 worker 的场景。
    """

    _WORKER = """
import sys
sys.path.insert(0, {root!r})
from app.config import Settings
from app.models import TicketIn
from app.services.tickets import TicketStore

store = TicketStore(Settings(ticket_store={path!r}))
t = store.create(TicketIn(title="跨进程故障描述", summary="跨进程现象", priority="P3"))
print(t.ticket_id)
"""

    def test_parallel_processes_never_duplicate_ids(self, tmp_path):
        import subprocess
        import sys

        root = str(Path(__file__).resolve().parents[1])
        target = tmp_path / "cross.json"
        script = tmp_path / "worker.py"
        script.write_text(
            self._WORKER.format(root=root, path=str(target)), encoding="utf-8"
        )

        n = 6
        procs = [
            subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(n)
        ]
        ids, errs = [], []
        for p in procs:
            out, err = p.communicate(timeout=90)
            if p.returncode != 0:
                errs.append(err.strip()[-300:])
            else:
                ids.append(out.strip())

        assert not errs, f"子进程失败：{errs}"
        assert len(ids) == n, ids
        assert len(set(ids)) == n, f"跨进程发出重复编号：{sorted(ids)}"

        # 落盘结果必须完整可解析（原子写的验证）
        rows = json.loads(target.read_text(encoding="utf-8"))
        assert len(rows) == n, f"落盘条数不符：{len(rows)} != {n}"

    def test_atomic_write_leaves_no_partial_file(self, tmp_path):
        """写入过程中不得出现半截 JSON：临时文件必须与数据文件分离。"""
        from app.services.tickets import TicketStore

        target = tmp_path / "atomic.json"
        store = TicketStore(Settings(ticket_store=target))
        store.create(TicketIn(title="原子写验证", summary="验证现象", priority="P3"))

        leftovers = [p.name for p in tmp_path.glob("*.tmp")]
        assert leftovers == [], f"残留临时文件：{leftovers}"
        assert json.loads(target.read_text(encoding="utf-8"))


def _create_one(client) -> str:
    """走 /v1/agent 建一张单（离线兜底也会真建），返回其 ticket_id。"""
    resp = client.post("/v1/agent", json={"question": "支付网关连不上，帮我建工单"})
    assert resp.status_code == 200
    assert resp.json()["intent"] == "create_ticket"
    # 建完从列表末尾把 id 取回来
    items = client.get("/v1/tickets").json()["items"]
    return items[-1]["ticket_id"]


class TestTickets:
    def test_created_ticket_appears_in_list(self, client):
        before = client.get("/v1/tickets").json()["total"]
        _create_one(client)
        after = client.get("/v1/tickets").json()
        assert after["total"] == before + 1
        assert after["items"] is not None

    def test_ticket_detail_found(self, client):
        tid = _create_one(client)
        resp = client.get(f"/v1/tickets/{tid}")
        assert resp.status_code == 200
        assert resp.json()["ticket_id"] == tid

    def test_ticket_detail_not_found(self, client):
        resp = client.get("/v1/tickets/TCK-9999")
        assert resp.status_code == 404


class TestClose:
    def test_close_flips_status(self, store):
        store.create(TicketIn(title="支付网关超时", summary="错误率上升", priority="P1"))
        assert store.close("TCK-0001") is True
        assert store.list()[0].status == "closed"

    def test_close_unknown_returns_false(self, store):
        assert store.close("TCK-9999") is False


class TestPersistence:
    def test_tickets_are_persisted(self, store):
        store.create(TicketIn(title="支付网关超时", summary="错误率上升", priority="P1"))
        settings = Settings(ticket_store=store.path)
        restarted = TicketStore(settings)
        assert restarted.count() == 1
        assert isinstance(restarted.list()[0].created_at, datetime)
