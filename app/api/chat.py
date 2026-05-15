"""Chat API — SSE streaming endpoint.

POST /api/v1/chat  →  text/event-stream

Uses the manual ReAct loop in ``chat_service.stream_chat()`` to achieve
token-level streaming identical to ChatGPT / Claude / Gemini.
"""

from __future__ import annotations

import uuid
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.engine import get_db
from app.models.user import User
from app.schemas.chat import ChatRequest
from app.services import chat_service, session_service

from app.storage.redis_client import acquire_lock, release_lock, get_redis

router = APIRouter(prefix="/chat", tags=["chat"])

class StopRequest(BaseModel):
    session_id: uuid.UUID


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
    session = await session_service.get_session(
        db, payload.session_id, user.id, payload.project_id
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    lock_key = f"session_lock:{payload.session_id}"
    if not await acquire_lock(lock_key, timeout=300):
        raise HTTPException(
            status_code=409, 
            detail="Dialog is generating. Please stop it first."
        )

    queue = asyncio.Queue()
    redis = await get_redis()
    await redis.delete(f"session:abort:{payload.session_id}")

    async def producer_wrapper():
        try:
            async for chunk in chat_service.stream_chat(
                session_id=str(payload.session_id),
                user_content=payload.content,
                user_id=str(user.id),
                plan_mode=payload.plan_mode,
                file_ids=[str(fid) for fid in payload.file_ids],
                project_id=str(payload.project_id) if payload.project_id else None,
            ):
                await queue.put(chunk)
            await queue.put(None)
        except Exception as e:
            await queue.put(e)
        finally:
            await release_lock(lock_key)

    asyncio.create_task(producer_wrapper())

    async def _consumer():
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    import json
                    error_payload = f"data: {json.dumps({'event': 'error', 'data': {'message': str(chunk)}}, ensure_ascii=False)}\n\n"
                    yield error_payload
                    break
                yield chunk
        except asyncio.CancelledError:
            # Client disconnected gracefully, background producer continues to finish the query.
            pass

    return StreamingResponse(
        _consumer(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/stop")
async def stop_chat(payload: StopRequest, user: User = Depends(get_current_user)):
    redis = await get_redis()
    await redis.set(f"session:abort:{payload.session_id}", "1", ex=600)
    return {"status": "ok"}
