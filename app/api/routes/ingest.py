"""文档入库、列举与删除。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_kb, require_auth
from app.models import DocumentIn, DocumentSummary, IngestRequest, IngestResponse, IngestResult
from app.services.document_loader import (
    infer_title,
    read_bytes,
    service_from_meta,
    tags_from_meta,
)
from app.services.knowledge import KnowledgeBase

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("", response_model=IngestResponse, summary="批量入库（JSON）")
def ingest_documents(
    payload: IngestRequest,
    kb: KnowledgeBase = Depends(get_kb),
    _claims: dict = Depends(require_auth),
) -> IngestResponse:
    return kb.ingest(payload.documents, replace=payload.replace)


@router.post("/upload", response_model=IngestResult, summary="上传文件入库（md / txt / pdf）")
async def upload_document(
    file: Annotated[UploadFile, File(description="待入库的文件")],
    service: Annotated[str | None, Form(description="所属服务，可选")] = None,
    tags: Annotated[str | None, Form(description="逗号分隔的标签，可选")] = None,
    kb: KnowledgeBase = Depends(get_kb),
    _claims: dict = Depends(require_auth),
) -> IngestResult:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件内容为空")

    filename = file.filename or "upload.txt"
    try:
        meta, body = read_bytes(filename, data)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc

    if not body.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="未能从文件中解析出文本内容",
        )

    doc = DocumentIn(
        title=infer_title(meta, body, fallback=Path(filename).stem),
        text=body,
        source=filename,
        service=service or service_from_meta(meta),
        tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else tags_from_meta(meta),
    )
    return kb.ingest([doc], replace=True).ingested[0]


@router.get("", response_model=list[DocumentSummary], summary="列出已入库文档")
def list_documents(
    kb: KnowledgeBase = Depends(get_kb),
    _claims: dict = Depends(require_auth),
) -> list[DocumentSummary]:
    return kb.list_documents()


@router.delete("/{doc_id}", summary="删除文档及其全部块")
def delete_document(
    doc_id: str,
    kb: KnowledgeBase = Depends(get_kb),
    _claims: dict = Depends(require_auth),
) -> dict:
    removed = kb.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"未找到文档 {doc_id}"
        )
    return {"doc_id": doc_id, "removed_chunks": removed}
