"""Project CRUD service."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, NotFoundError
from app.models.project import Project


async def list_projects(db: AsyncSession, user_id: uuid.UUID) -> list[Project]:
    result = await db.execute(
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(Project.updated_at.desc())
    )
    return list(result.scalars().all())


async def create_project(
    db: AsyncSession,
    user_id: uuid.UUID,
    name: str,
    description: str | None = None,
) -> Project:
    project = Project(user_id=user_id, name=name, description=description)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


async def _get_owned(
    db: AsyncSession, project_id: uuid.UUID, user_id: uuid.UUID
) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    if project.user_id != user_id:
        raise ForbiddenError("You do not own this project")
    return project


async def get_project(
    db: AsyncSession, project_id: uuid.UUID, user_id: uuid.UUID
) -> Project:
    return await _get_owned(db, project_id, user_id)


async def update_project(
    db: AsyncSession,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Project:
    project = await _get_owned(db, project_id, user_id)
    if name is not None:
        project.name = name
    if description is not None:
        project.description = description
    await db.commit()
    await db.refresh(project)
    return project


async def delete_project(
    db: AsyncSession, project_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    project = await _get_owned(db, project_id, user_id)
    await db.delete(project)
    await db.commit()


async def get_default_project(
    db: AsyncSession, user_id: uuid.UUID
) -> Project:
    """Return the user's first project, creating a Default Project if none exists.

    Used as a safety net for legacy code paths that don't yet pass project_id.
    """
    result = await db.execute(
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(Project.created_at.asc())
        .limit(1)
    )
    project = result.scalar_one_or_none()
    if project is not None:
        return project
    return await create_project(db, user_id, "Default Project")
