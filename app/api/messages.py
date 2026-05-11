"""Messages API — read conversation history."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.engine import get_db
from app.models.session_file import SessionFile
from app.models.user import User
from app.schemas.message import MessageAttachment, MessageResponse
from app.services import session_service
from app.storage.manager import load_messages

router = APIRouter(prefix="/projects/{project_id}/sessions", tags=["messages"])


@router.get(
    "/{session_id}/messages",
    response_model=list[MessageResponse],
)
async def get_session_messages(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MessageResponse]:
    session = await session_service.get_session(db, session_id, user.id, project_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    raw = await load_messages(str(session_id), limit=limit)

    # Collect every file_id referenced by the messages we're about to
    # return so we can resolve attachment metadata in a single query.
    referenced_ids: set[uuid.UUID] = set()
    for m in raw:
        for fid in m.get("file_ids") or []:
            try:
                referenced_ids.add(uuid.UUID(str(fid)))
            except (ValueError, TypeError):
                continue

    files_by_id: dict[uuid.UUID, SessionFile] = {}
    if referenced_ids:
        result = await db.execute(
            select(SessionFile).where(SessionFile.id.in_(referenced_ids))
        )
        for row in result.scalars().all():
            files_by_id[row.id] = row

    out: list[MessageResponse] = []
    for m in raw:
        file_ids: list[str] = [str(fid) for fid in (m.get("file_ids") or [])]
        attachments: list[MessageAttachment] = []
        for fid_str in file_ids:
            try:
                rec = files_by_id.get(uuid.UUID(fid_str))
            except (ValueError, TypeError):
                rec = None
            if rec is None:
                continue
            attachments.append(
                MessageAttachment(
                    id=str(rec.id),
                    filename=rec.filename,
                    original_filename=rec.original_filename,
                    mime_type=rec.mime_type,
                    size=rec.size,
                    download_url=(
                        f"/api/v1/projects/{rec.project_id}/sessions/"
                        f"{rec.session_id}/files/{rec.id}/download"
                    ),
                )
            )
        out.append(
            MessageResponse(
                id=str(m.get("id") or ""),
                role=str(m.get("role") or ""),
                content=str(m.get("content") or ""),
                metadata=m.get("metadata"),
                token_count=m.get("token_count"),
                created_at=m.get("ts") or m.get("created_at"),
                file_ids=file_ids,
                attachments=attachments,
            )
        )
    return out
