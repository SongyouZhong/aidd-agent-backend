"""Task management API.

Provides endpoints for controlling background tasks:
  DELETE /api/v1/tasks/{task_id}  — cancel a running or pending task
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.models.user import User
from app.services import task_registry
from app.services.background_runner import background_runner

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_task(
    task_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
) -> None:
    """Cancel a background task owned by the current user.

    Stops the asyncio task (if still running) and transitions the task
    record to 'cancelled', broadcasting a ``task_cancelled`` event on
    the session's event channel.
    """
    task = await task_registry.get(str(task_id))
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    if str(task.user_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your task")

    if task.status in task_registry.TERMINAL_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task already in terminal state: {task.status}",
        )

    await background_runner.cancel(str(task_id))
    await task_registry.cancel(str(task_id))
