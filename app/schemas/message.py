"""Message API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MessageAttachment(BaseModel):
    """Lightweight file attachment metadata returned alongside a message.

    Includes everything the frontend needs to render an attachment chip
    and (for ``text/markdown`` / ``application/json``) auto-open it in
    the report viewer side panel — no extra round-trip required.
    """

    id: str
    filename: str
    original_filename: str
    mime_type: str
    size: int
    download_url: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict[str, Any] | None = None
    token_count: int | None = None
    created_at: str | None = None  # ISO 8601 ts from JSONL
    file_ids: list[str] = []
    attachments: list[MessageAttachment] = []
