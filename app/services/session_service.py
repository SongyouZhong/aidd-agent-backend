"""Session CRUD service."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, NotFoundError
from app.models.session import Session
from app.services import project_service
from app.storage.s3 import session_prefix


async def list_sessions(
    db: AsyncSession,
    user_id: uuid.UUID,
    project_id: uuid.UUID,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[Session]:
    # Validates project ownership.
    await project_service.get_project(db, project_id, user_id)
    result = await db.execute(
        select(Session)
        .where(Session.user_id == user_id, Session.project_id == project_id)
        .order_by(Session.is_pinned.desc(), Session.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    project_id: uuid.UUID,
    title: str | None = None,
) -> Session:
    # Validates project ownership.
    await project_service.get_project(db, project_id, user_id)
    session = Session(
        user_id=user_id,
        project_id=project_id,
        title=title or "新对话",
    )
    db.add(session)
    await db.flush()
    session.s3_prefix = session_prefix(str(session.id))
    await db.commit()
    await db.refresh(session)
    return session


async def _get_owned(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    project_id: uuid.UUID | None = None,
) -> Session:
    session = await db.get(Session, session_id)
    if session is None:
        raise NotFoundError("Session not found")
    if session.user_id != user_id:
        raise ForbiddenError("You do not own this session")
    if project_id is not None and session.project_id != project_id:
        raise NotFoundError("Session not found in this project")
    return session


async def update_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    project_id: uuid.UUID,
    *,
    title: str | None = None,
    is_pinned: bool | None = None,
) -> Session:
    session = await _get_owned(db, session_id, user_id, project_id)
    if title is not None:
        session.title = title
    if is_pinned is not None:
        session.is_pinned = is_pinned
    await db.commit()
    await db.refresh(session)
    return session


# Back-compat shim used by older code paths.
async def rename_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    title: str,
    project_id: uuid.UUID | None = None,
) -> Session:
    session = await _get_owned(db, session_id, user_id, project_id)
    session.title = title
    await db.commit()
    await db.refresh(session)
    return session


async def delete_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    project_id: uuid.UUID | None = None,
) -> None:
    session = await _get_owned(db, session_id, user_id, project_id)
    await db.delete(session)
    await db.commit()


async def get_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    project_id: uuid.UUID | None = None,
) -> Session | None:
    """Return session if owned by user (and in project, if specified), else None."""
    session = await db.get(Session, session_id)
    if session is None or session.user_id != user_id:
        return None
    if project_id is not None and session.project_id != project_id:
        return None
    return session
