"""Chat API — SSE streaming endpoint.

POST /api/v1/chat  →  text/event-stream

Uses the manual ReAct loop in ``chat_service.stream_chat()`` to achieve
token-level streaming identical to ChatGPT / Claude / Gemini.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.engine import get_db
from app.models.user import User
from app.schemas.chat import ChatRequest
from app.services import chat_service, session_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a message and receive a streaming SSE response.

    The response uses ``Content-Type: text/event-stream``.
    Each event is ``data: {json}\\n\\n``.  Stream ends with ``data: [DONE]\\n\\n``.

    To stop generation, the client aborts the fetch request via AbortController.
    """
    session = await session_service.get_session(db, payload.session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        chat_service.stream_chat(
            session_id=str(payload.session_id),
            user_content=payload.content,
            user_id=str(user.id),
            plan_mode=payload.plan_mode,
            file_ids=[str(fid) for fid in payload.file_ids],
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
