"""Session CRUD API — nested under /projects/{project_id}/sessions."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.engine import get_db
from app.models.user import User
from app.schemas.session import SessionCreate, SessionResponse, SessionUpdate
from app.services import session_service
from app.services import task_registry
from app.storage.manager import drop_session_cache

router = APIRouter(prefix="/projects/{project_id}/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    project_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[SessionResponse]:
    sessions = await session_service.list_sessions(
        db, user.id, project_id, limit=limit, offset=offset
    )
    return [SessionResponse.model_validate(s) for s in sessions]


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    project_id: uuid.UUID,
    payload: SessionCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = await session_service.create_session(
        db, user.id, project_id, payload.title
    )
    return SessionResponse.model_validate(session)


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    payload: SessionUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = await session_service.update_session(
        db,
        session_id,
        user.id,
        project_id,
        title=payload.title,
        is_pinned=payload.is_pinned,
    )
    return SessionResponse.model_validate(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    await session_service.delete_session(db, session_id, user.id, project_id)
    await drop_session_cache(str(session_id))


@router.get("/{session_id}/active-tasks")
async def get_active_tasks(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return tasks for this session — active plus terminal tasks from the last 24 h.

    Used by the frontend on page load to hydrate the right-panel task list
    without waiting for the first SSE event.
    """
    session = await session_service.get_session(db, session_id, user.id, project_id)
    if session is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")

    tasks = await task_registry.list_by_session_recent(str(session_id))
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "kind": t.kind,
                "status": t.status,
                "percent": t.percent,
                "phase": t.phase,
                "desc": t.desc,
                "target": t.target,
                "started_at": t.started_at,
                "finished_at": t.finished_at,
                "result": t.result,
                "error": t.error,
            }
            for t in tasks
        ]
    }
