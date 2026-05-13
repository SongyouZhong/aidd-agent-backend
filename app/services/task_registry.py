"""Background task state registry backed by Redis.

Data layout (all keys prefixed for namespacing):
  task:{task_id}               Hash   — full task state (TTL 7 days)
  session_tasks:{session_id}   Set    — task_ids belonging to a session
  session_events:{session_id}  channel — Pub/Sub real-time event stream

This module is the single source of truth for task status. All callers
(background worker, HTTP endpoints, SSE event stream) read/write through
these helpers rather than touching Redis directly.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.storage.redis_client import get_redis, publish

logger = logging.getLogger(__name__)

_TASK_TTL = 7 * 24 * 3600        # 7 days in seconds
_SESSION_TASKS_TTL = 7 * 24 * 3600

TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TaskState:
    task_id: str
    session_id: str
    user_id: str
    project_id: str
    kind: str           # e.g. "target_discovery"
    status: str         # pending / running / succeeded / failed / cancelled
    percent: int        # 0–100
    phase: str          # pipeline phase name
    desc: str           # human-readable status description
    target: str         # the subject of the task (gene symbol etc.)
    started_at: str     # ISO-8601
    tool_call_id: str = ""
    finished_at: str | None = None
    result: dict | None = None   # stored as JSON in Redis
    error: str | None = None


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _task_key(task_id: str) -> str:
    return f"task:{task_id}"


def _session_tasks_key(session_id: str) -> str:
    return f"session_tasks:{session_id}"


def _events_channel(session_id: str) -> str:
    return f"session_events:{session_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Redis serialisation helpers
# ---------------------------------------------------------------------------

def _to_hash(state: TaskState) -> dict[str, str]:
    """Flatten a TaskState into a Redis-compatible string map."""
    raw = asdict(state)
    out: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def _from_hash(data: dict[str, str]) -> TaskState:
    """Reconstruct a TaskState from a Redis HGETALL result."""
    result_raw = data.get("result") or ""
    result: dict | None = None
    if result_raw:
        try:
            result = json.loads(result_raw)
        except Exception:
            pass
    try:
        percent = int(data.get("percent", "0"))
    except (ValueError, TypeError):
        percent = 0
    return TaskState(
        task_id=data.get("task_id", ""),
        session_id=data.get("session_id", ""),
        user_id=data.get("user_id", ""),
        project_id=data.get("project_id", ""),
        kind=data.get("kind", ""),
        status=data.get("status", ""),
        percent=percent,
        phase=data.get("phase", ""),
        desc=data.get("desc", ""),
        target=data.get("target", ""),
        started_at=data.get("started_at", ""),
        tool_call_id=data.get("tool_call_id", ""),
        finished_at=data.get("finished_at") or None,
        result=result,
        error=data.get("error") or None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_task(
    *,
    session_id: str,
    user_id: str,
    project_id: str | None,
    kind: str,
    target: str,
) -> str:
    """Create a new task record in Redis and return the task_id."""
    task_id = str(uuid.uuid4())
    state = TaskState(
        task_id=task_id,
        session_id=session_id,
        user_id=user_id,
        project_id=project_id or "",
        kind=kind,
        status="pending",
        percent=0,
        phase="",
        desc="Accepted, waiting to start...",
        target=target,
        started_at=_now_iso(),
    )
    r = await get_redis()
    pipe = r.pipeline()
    pipe.hset(_task_key(task_id), mapping=_to_hash(state))
    pipe.expire(_task_key(task_id), _TASK_TTL)
    pipe.sadd(_session_tasks_key(session_id), task_id)
    pipe.expire(_session_tasks_key(session_id), _SESSION_TASKS_TTL)
    await pipe.execute()
    logger.info("Created task %s kind=%s target=%s session=%s", task_id, kind, target, session_id)
    return task_id


async def set_tool_call_id(task_id: str, tool_call_id: str) -> None:
    """Associate the originating LLM tool_call_id with this task."""
    r = await get_redis()
    await r.hset(_task_key(task_id), "tool_call_id", tool_call_id)


async def update_progress(
    task_id: str,
    percent: int,
    phase: str,
    desc: str,
) -> None:
    """Update task progress and broadcast a task_progress event."""
    r = await get_redis()
    state_data = await r.hgetall(_task_key(task_id))
    if not state_data:
        logger.warning("update_progress: task %s not found in Redis", task_id)
        return
    session_id = state_data.get("session_id", "")
    await r.hset(
        _task_key(task_id),
        mapping={"status": "running", "percent": str(percent), "phase": phase, "desc": desc},
    )
    event = {
        "type": "task_progress",
        "task": {
            "task_id": task_id,
            "percent": percent,
            "phase": phase,
            "desc": desc,
            "target": state_data.get("target", ""),
            "status": "running",
        },
    }
    await publish(_events_channel(session_id), json.dumps(event, ensure_ascii=False))


async def complete(task_id: str, result: dict[str, Any]) -> None:
    """Mark the task as succeeded and broadcast task_completed."""
    r = await get_redis()
    state_data = await r.hgetall(_task_key(task_id))
    if not state_data:
        logger.warning("complete: task %s not found in Redis", task_id)
        return
    session_id = state_data.get("session_id", "")
    now = _now_iso()
    await r.hset(
        _task_key(task_id),
        mapping={
            "status": "succeeded",
            "percent": "100",
            "phase": "done",
            "desc": "Complete.",
            "finished_at": now,
            "result": json.dumps(result, ensure_ascii=False),
        },
    )
    event = {
        "type": "task_completed",
        "task": {
            "task_id": task_id,
            "result": result,
            "status": "succeeded",
            "percent": 100,
            "target": state_data.get("target", ""),
        },
    }
    await publish(_events_channel(session_id), json.dumps(event, ensure_ascii=False))


async def fail(task_id: str, error: str) -> None:
    """Mark the task as failed and broadcast task_failed."""
    r = await get_redis()
    state_data = await r.hgetall(_task_key(task_id))
    if not state_data:
        logger.warning("fail: task %s not found in Redis", task_id)
        return
    session_id = state_data.get("session_id", "")
    now = _now_iso()
    await r.hset(
        _task_key(task_id),
        mapping={"status": "failed", "finished_at": now, "error": error},
    )
    event = {
        "type": "task_failed",
        "task": {
            "task_id": task_id,
            "error": error,
            "status": "failed",
            "target": state_data.get("target", ""),
        },
    }
    await publish(_events_channel(session_id), json.dumps(event, ensure_ascii=False))


async def cancel(task_id: str) -> None:
    """Mark the task as cancelled and broadcast task_cancelled."""
    r = await get_redis()
    state_data = await r.hgetall(_task_key(task_id))
    if not state_data:
        logger.warning("cancel: task %s not found in Redis", task_id)
        return
    session_id = state_data.get("session_id", "")
    now = _now_iso()
    await r.hset(
        _task_key(task_id),
        mapping={"status": "cancelled", "finished_at": now},
    )
    event = {
        "type": "task_cancelled",
        "task": {
            "task_id": task_id,
            "status": "cancelled",
            "target": state_data.get("target", ""),
        },
    }
    await publish(_events_channel(session_id), json.dumps(event, ensure_ascii=False))


async def get(task_id: str) -> TaskState | None:
    """Fetch a task by ID. Returns None if not found or expired."""
    r = await get_redis()
    data = await r.hgetall(_task_key(task_id))
    if not data:
        return None
    return _from_hash(data)


async def list_by_session(
    session_id: str,
    *,
    include_terminal: bool = False,
) -> list[TaskState]:
    """Return tasks for a session. By default only non-terminal tasks."""
    r = await get_redis()
    task_ids: set[str] = await r.smembers(_session_tasks_key(session_id))
    tasks: list[TaskState] = []
    for tid in task_ids:
        state = await get(tid)
        if state is None:
            continue
        if not include_terminal and state.status in TERMINAL_STATUSES:
            continue
        tasks.append(state)
    return tasks


async def list_by_session_recent(session_id: str) -> list[TaskState]:
    """Return active tasks + terminal tasks finished within the last 24 hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    r = await get_redis()
    task_ids: set[str] = await r.smembers(_session_tasks_key(session_id))
    tasks: list[TaskState] = []
    for tid in task_ids:
        state = await get(tid)
        if state is None:
            continue
        if state.status not in TERMINAL_STATUSES:
            tasks.append(state)
        elif state.finished_at:
            try:
                finished = datetime.fromisoformat(state.finished_at.replace("Z", "+00:00"))
                if finished >= cutoff:
                    tasks.append(state)
            except Exception:
                pass
    return tasks


async def publish_event(session_id: str, event: dict[str, Any]) -> None:
    """Publish an arbitrary event dict to the session's event channel."""
    await publish(_events_channel(session_id), json.dumps(event, ensure_ascii=False))


async def reap_stale_tasks() -> None:
    """At startup: mark in-flight tasks as failed(process_restart).

    Any task still in 'running' or 'pending' state from a previous process
    will never complete, so we transition them to 'failed' and broadcast.
    """
    r = await get_redis()
    async for key in r.scan_iter("task:*"):
        data = await r.hgetall(key)
        if not data:
            continue
        status = data.get("status", "")
        if status not in ("running", "pending"):
            continue
        task_id = data.get("task_id", "")
        session_id = data.get("session_id", "")
        if not task_id:
            continue
        now = _now_iso()
        await r.hset(key, mapping={"status": "failed", "finished_at": now, "error": "process_restart"})
        logger.info("Reaped stale task %s (was %s)", task_id, status)
        if session_id:
            event = {
                "type": "task_failed",
                "task": {
                    "task_id": task_id,
                    "error": "process_restart",
                    "status": "failed",
                    "target": data.get("target", ""),
                },
            }
            await publish(_events_channel(session_id), json.dumps(event, ensure_ascii=False))
