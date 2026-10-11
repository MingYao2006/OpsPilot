"""工单持久化测试（全部落在 tmp_path，不碰 work/state 真数据）。"""

from __future__ import annotations

from datetime import datetime

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
