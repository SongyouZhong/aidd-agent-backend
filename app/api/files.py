"""File upload API — nested under /projects/{project_id}/sessions/{session_id}/files."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.engine import get_db
from app.models.user import User
from app.schemas.file import FileResponse
from app.services import file_service, session_service

router = APIRouter(prefix="/projects/{project_id}/sessions", tags=["files"])


@router.post(
    "/{session_id}/files",
    response_model=FileResponse,
    status_code=201,
)
async def upload_file(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    file: UploadFile = File(...),
    description: str | None = Form(default=None, max_length=500),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await session_service.get_session(db, session_id, user.id, project_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    mime = file.content_type or "application/octet-stream"

    record = await file_service.upload_file(
        db,
        project_id=project_id,
        session_id=session_id,
        user_id=user.id,
        filename=file.filename,
        content=content,
        mime_type=mime,
        description=description,
    )
    return record


@router.get(
    "/{session_id}/files",
    response_model=list[FileResponse],
)
async def list_files(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await session_service.get_session(db, session_id, user.id, project_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await file_service.list_files(db, session_id)


@router.get(
    "/{session_id}/files/{file_id}",
    response_model=FileResponse,
)
async def get_file(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    file_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await file_service.get_file(db, file_id, user.id)


@router.get("/{session_id}/files/{file_id}/download")
async def download_file(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    file_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream file content directly from S3 through the backend.

    Previously this returned a 302 redirect to a presigned S3 URL, but
    that breaks when the browser cannot reach the S3 endpoint (e.g.
    ``localhost:8333`` is only reachable on the server, not from a remote
    browser).  Proxying the content avoids this network topology issue.
    """
    from fastapi.responses import Response

    record = await file_service.get_file(db, file_id, user.id)
    content = await file_service.get_file_content(db, file_id, user.id)
    if content is None:
        raise HTTPException(status_code=404, detail="File content not found in S3")

    return Response(
        content=content,
        media_type=record.mime_type,
        headers={
            "Content-Disposition": f'inline; filename="{record.original_filename}"',
        },
    )


@router.get("/{session_id}/files/{file_id}/presigned-url")
async def get_presigned_url(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    file_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the pre-signed S3 URL as JSON so the frontend can use it
    for browser-based downloads / new-tab opens without needing to attach
    an Authorization header to the final redirect (which plain <a href>
    cannot do)."""
    url = await file_service.get_download_url(db, file_id, user.id)
    return {"url": url}


@router.get("/{session_id}/files/{file_id}/pdf")
async def download_pdf(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    file_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Render a Markdown file as PDF via the remark-renderer microservice.

    Flow: read MD from S3 → POST to remark-renderer → return PDF bytes.
    """
    import httpx
    from fastapi.responses import Response
    from app.core.config import settings

    record = await file_service.get_file(db, file_id, user.id)
    if not record.mime_type.startswith("text/"):
        raise HTTPException(status_code=400, detail="Only text files can be converted to PDF")

    content = await file_service.get_file_content(db, file_id, user.id)
    if content is None:
        raise HTTPException(status_code=404, detail="File content not found in S3")

    md_text = content.decode("utf-8")
    title = record.original_filename.replace(".md", "")

    # Call remark-renderer
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.REMARK_RENDERER_URL}/render",
                json={"markdown": md_text, "title": title},
            )
        if resp.status_code != 200:
            detail = resp.json().get("error", "Renderer error") if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:200]
            raise HTTPException(status_code=502, detail=f"PDF renderer failed: {detail}")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="PDF renderer service unavailable")

    pdf_filename = title + ".pdf"
    return Response(
        content=resp.content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{pdf_filename}"',
        },
    )


@router.delete(
    "/{session_id}/files/{file_id}",
    status_code=204,
)
async def delete_file(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    file_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await file_service.delete_file(db, file_id, user.id)

