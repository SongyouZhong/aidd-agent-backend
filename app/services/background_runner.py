"""In-process background task runner.

Wraps asyncio.create_task behind an interface that can be replaced by
arq/Celery in the future without changing call sites.

Usage::

    from app.services.background_runner import background_runner

    await background_runner.submit(my_coroutine(), task_id="abc-123")
    await background_runner.cancel("abc-123")

The singleton ``background_runner`` is shut down in ``app.main.lifespan``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundRunner:
    """Manages a set of named asyncio tasks."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    async def submit(self, coro: Coroutine[Any, Any, Any], *, task_id: str) -> None:
        """Schedule *coro* as a background task identified by *task_id*."""
        task: asyncio.Task[Any] = asyncio.create_task(coro)
        task.set_name(task_id)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.info("Submitted background task %s", task_id)

    async def cancel(self, task_id: str) -> bool:
        """Cancel the task with the given *task_id*. Returns True if found."""
        for t in list(self._tasks):
            if t.get_name() == task_id:
                t.cancel()
                logger.info("Cancelled background task %s", task_id)
                return True
        return False

    async def shutdown(self) -> None:
        """Cancel all running tasks and wait for them to finish."""
        pending = list(self._tasks)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        logger.info("BackgroundRunner shutdown (%d task(s) cancelled)", len(pending))


# Module-level singleton — imported by tools and API handlers.
background_runner = BackgroundRunner()
