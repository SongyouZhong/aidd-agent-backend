"""Persist deep-research target reports as session files.

Used by the ``run_target_discovery`` tool to write the full
``TargetReport`` JSON to S3 and create a ``SessionFile`` row so the
frontend can download it and the conversation can later reload it via
``_load_file_context``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from fastapi import HTTPException

from app.db.engine import AsyncSessionLocal
from app.models.session import Session as SessionRow
from app.models.session_file import SessionFile
from app.services import file_service

logger = logging.getLogger(__name__)


def _safe_filename(target_query: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", target_query).strip("_")[:60] or "target"
    return f"{slug}_target_report.json"


async def save_report_as_session_file(
    *,
    session_id: str,
    user_id: str,
    project_id: str | None,
    target_query: str,
    report: dict[str, Any],
) -> SessionFile:
    """Write report JSON to S3 + create SessionFile row. Returns the row.

    ``project_id`` may be None — in that case we look it up from the
    ``sessions`` table so the file is correctly scoped.
    """
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    content_bytes = payload.encode("utf-8")
    filename = _safe_filename(target_query)

    async with AsyncSessionLocal() as db:
        # Resolve project_id from session if not provided.
        if project_id is None:
            row = await db.get(SessionRow, uuid.UUID(session_id))
            if row is None:
                raise RuntimeError(f"Session {session_id} not found")
            project_uuid = row.project_id
        else:
            project_uuid = uuid.UUID(project_id)

        try:
            record = await file_service.upload_file(
                db,
                project_id=project_uuid,
                session_id=uuid.UUID(session_id),
                user_id=uuid.UUID(user_id),
                filename=filename,
                content=content_bytes,
                mime_type="application/json",
                description=f"Deep-research target report for: {target_query}",
            )
        except HTTPException as exc:
            # Surface upload errors as runtime errors — the calling tool
            # will catch and downgrade to a notes entry.
            raise RuntimeError(f"upload_file failed: {exc.detail}") from exc

    logger.info(
        "Saved target-discovery report to session file: file_id=%s session=%s",
        record.id,
        session_id,
    )
    return record
