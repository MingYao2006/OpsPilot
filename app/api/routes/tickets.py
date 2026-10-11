"""工单查询接口：/v1/tickets 列表 + /v1/tickets/{ticket_id} 详情（只读，构成"建单→查单"闭环）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_container, require_auth
from app.container import Container
from app.models import Ticket, TicketListResponse

router = APIRouter(prefix="/v1", tags=["tickets"])


@router.get("/tickets", response_model=TicketListResponse, summary="工单列表")
def list_tickets(
    container: Container = Depends(get_container),
    _claims: dict = Depends(require_auth),
) -> TicketListResponse:
    items = container.tickets.list()
    return TicketListResponse(total=len(items), items=items)


@router.get("/tickets/{ticket_id}", response_model=Ticket, summary="工单详情")
def get_ticket(
    ticket_id: str,
    container: Container = Depends(get_container),
    _claims: dict = Depends(require_auth),
) -> Ticket:
    ticket = container.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"工单不存在：{ticket_id}",
        )
    return ticket