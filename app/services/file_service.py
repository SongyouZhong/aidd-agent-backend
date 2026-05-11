"""File upload/download/delete service.

Files are stored in SeaweedFS (S3); metadata is in PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, NotFoundError
from app.models.session_file import SessionFile
from app.schemas.file import ALLOWED_MIME_TYPES, MAX_FILE_SIZE, MAX_FILES_PER_SESSION
from app.storage.s3 import file_key, s3_storage

logger = logging.getLogger(__name__)


async def upload_file(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    filename: str,
    content: bytes,
    mime_type: str,
    description: str | None = None,
) -> SessionFile:
    """Upload a file to S3 and create a metadata record in PostgreSQL."""

    # --- validate MIME type ---
    if mime_type not in ALLOWED_MIME_TYPES:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {mime_type}. Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )

    # --- validate size ---
    if len(content) > MAX_FILE_SIZE:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)} MB",
        )

    # --- validate per-session file count ---
    count = await db.scalar(
        select(func.count()).select_from(SessionFile).where(
            SessionFile.session_id == session_id
        )
    )
    if count is not None and count >= MAX_FILES_PER_SESSION:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Maximum {MAX_FILES_PER_SESSION} files per session",
        )

    # --- upload to S3 ---
    fid = uuid.uuid4()
    safe_name = filename.replace("/", "_").replace("\\", "_")
    s3_key = file_key(str(session_id), str(fid), safe_name)

    await s3_storage.put_object(s3_key, content, content_type=mime_type)

    # --- create DB record ---
    record = SessionFile(
        id=fid,
        project_id=project_id,
        session_id=session_id,
        user_id=user_id,
        filename=safe_name,
        original_filename=filename,
        mime_type=mime_type,
        size=len(content),
        description=description,
        s3_key=s3_key,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def list_files(
    db: AsyncSession, session_id: uuid.UUID
) -> list[SessionFile]:
    result = await db.execute(
        select(SessionFile)
        .where(SessionFile.session_id == session_id)
        .order_by(SessionFile.created_at.desc())
    )
    return list(result.scalars().all())


async def get_file(
    db: AsyncSession, file_id: uuid.UUID, user_id: uuid.UUID
) -> SessionFile:
    record = await db.get(SessionFile, file_id)
    if record is None:
        raise NotFoundError("File not found")
    if record.user_id != user_id:
        raise ForbiddenError("You do not own this file")
    return record


async def delete_file(
    db: AsyncSession, file_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    record = await get_file(db, file_id, user_id)
    # Delete from S3
    try:
        await s3_storage.delete_object(record.s3_key)
    except Exception:
        logger.warning("Failed to delete S3 object %s", record.s3_key)
    # Delete from DB
    await db.delete(record)
    await db.commit()


async def get_download_url(
    db: AsyncSession, file_id: uuid.UUID, user_id: uuid.UUID
) -> str:
    record = await get_file(db, file_id, user_id)
    return await s3_storage.presigned_get_url(record.s3_key, expires_in=600)
