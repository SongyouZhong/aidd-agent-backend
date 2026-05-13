"""Session event stream API.

Provides a persistent SSE channel that pushes real-time task progress
and message notifications to clients:

  GET /api/v1/projects/{project_id}/sessions/{session_id}/events

Clients subscribe once per session (using the same JWT-auth fetch +
ReadableStream pattern as the chat endpoint) and receive:
  • task_progress   — incremental pipeline progress (percent/phase/desc)
  • task_completed  — task finished, result available
  • task_failed     — task errored
  • task_cancelled  — task was cancelled
  • message_appended — a new assistant message was persisted (triggers
                       the frontend to invalidate the messages query)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.models.user import User
from app.services import session_service, task_registry
from app.storage.redis_client import get_pubsub
from app.db.engine import get_db
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects/{project_id}/sessions/{session_id}",
    tags=["events"],
)

_KEEPALIVE_INTERVAL = 15  # seconds between SSE keep-alive comments


@router.get("/events")
async def session_events(
    project_id: uuid.UUID,
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Subscribe to real-time task/message events for a session.

    Returns a ``text/event-stream`` response. Events are JSON objects;
    one per SSE ``data:`` line.  A keep-alive comment is sent every
    15 seconds to prevent proxy timeouts.
    """
    # Verify the session belongs to this user/project.
    session = await session_service.get_session(db, session_id, current_user.id, project_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    channel = f"session_events:{session_id}"

    async def _event_stream():
        pubsub = await get_pubsub()
        await pubsub.subscribe(channel)
        logger.info("SSE client subscribed to %s", channel)

        # Use a dedicated reader task + asyncio.Queue so that:
        # 1. pubsub.listen() runs in an uninterrupted async task (safe socket reads)
        # 2. The SSE generator reads from the queue with a timeout for keep-alives.
        #    Cancelling asyncio.Queue.get() is always safe (no socket state corruption).
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _reader() -> None:
            try:
                async for message in pubsub.listen():
                    if message and message.get("type") == "message":
                        await queue.put(message["data"])
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("PubSub reader error on %s: %s", channel, exc)
            finally:
                await queue.put(None)  # signal EOF to the generator

        reader_task = asyncio.create_task(_reader())

        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        queue.get(), timeout=_KEEPALIVE_INTERVAL
                    )
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                if data is None:
                    break  # reader exited
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            logger.info("SSE client disconnected from %s", channel)
        finally:
            reader_task.cancel()
            try:
                await asyncio.wait_for(reader_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception:
                pass

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
