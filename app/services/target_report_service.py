"""Persist deep-research target reports as session files.

Used by the ``run_target_discovery`` tool to write the full
``TargetReport`` to S3 as both a machine-readable JSON file and a
human-readable Markdown document, then create ``SessionFile`` rows so
the frontend can list/download/preview them and the conversation can
later reload them via ``_load_file_context``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.db.engine import AsyncSessionLocal
from app.models.session import Session as SessionRow
from app.models.session_file import SessionFile
from app.services import file_service
from app.services.report_renderer import render_target_report_md

logger = logging.getLogger(__name__)


def _safe_slug(target_query: str) -> str:
    return re.sub(r"[^\w\-]+", "_", target_query).strip("_")[:60] or "target"


@dataclass
class SavedReportFiles:
    """Pair of SessionFile rows produced for a single deep-research run."""

    json_record: SessionFile
    md_record: SessionFile


async def save_report_as_session_file(
    *,
    session_id: str,
    user_id: str,
    project_id: str | None,
    target_query: str,
    report: dict[str, Any],
    language: str = "English",
) -> SavedReportFiles:
    """Persist the report as TWO session files (JSON + MD). Returns both.

    The MD file is the user-facing artifact rendered by
    ``report_renderer.render_target_report_md``; the JSON file is the
    raw structured payload for downstream programmatic use.

    ``project_id`` may be None — in that case we look it up from the
    ``sessions`` table so the file is correctly scoped.
    """
    slug = _safe_slug(target_query)
    json_payload = json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    try:
        md_payload = render_target_report_md(report, target_query, language=language).encode("utf-8")
    except Exception as exc:
        # Renderer failure shouldn't block JSON persistence — fall back
        # to a minimal MD stub so the user can still see something.
        logger.warning("render_target_report_md failed (%s); using stub", exc)
        md_payload = (
            f"# {target_query} Target Discovery Report\n\n"
            f"_(Markdown rendering failed: {exc!r}. See JSON file for raw data.)_\n"
        ).encode("utf-8")

    json_filename = f"{slug}_target_report.json"
    md_filename = f"{slug}_target_report.md"

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
            md_record = await file_service.upload_file(
                db,
                project_id=project_uuid,
                session_id=uuid.UUID(session_id),
                user_id=uuid.UUID(user_id),
                filename=md_filename,
                content=md_payload,
                mime_type="text/markdown",
                description=f"Deep-research target report (Markdown) for: {target_query}",
            )
            json_record = await file_service.upload_file(
                db,
                project_id=project_uuid,
                session_id=uuid.UUID(session_id),
                user_id=uuid.UUID(user_id),
                filename=json_filename,
                content=json_payload,
                mime_type="application/json",
                description=f"Deep-research target report (raw JSON) for: {target_query}",
            )
        except HTTPException as exc:
            raise RuntimeError(f"upload_file failed: {exc.detail}") from exc

    logger.info(
        "Saved target-discovery report: md=%s json=%s session=%s",
        md_record.id,
        json_record.id,
        session_id,
    )
    return SavedReportFiles(json_record=json_record, md_record=md_record)
