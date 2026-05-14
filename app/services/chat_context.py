"""Per-request context for chat tools.

Tools invoked inside ``chat_service.stream_chat`` need access to the
current session/user/project IDs (e.g. to persist a report as a session
file) and an optional progress callback for long-running operations to
push SSE events back to the client.

We use ``contextvars`` so that the tool functions—called via LangChain's
``invoke``/``ainvoke`` wrappers—can pick up the context without changing
the registered ``@tool`` signature.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass
class ChatRequestContext:
    session_id: str
    user_id: str
    project_id: Optional[str] = None
    language: str = "English"


# Per-request context (set in chat_service.stream_chat entry).
current_chat_context: ContextVar[Optional[ChatRequestContext]] = ContextVar(
    "current_chat_context", default=None
)

# Optional async progress callback. Tools may call this to emit
# intermediate progress; chat_service forwards them as SSE events.
# Signature: await progress_callback(event_type: str, payload: dict) -> None
ProgressCallback = Callable[[str, dict], Awaitable[None]]
progress_callback: ContextVar[Optional[ProgressCallback]] = ContextVar(
    "progress_callback", default=None
)


# Re-entrancy guard: set when a deep-research tool is currently running
# in this context. Prevents the LLM from launching a second concurrent
# pipeline within the same chat turn.
deep_research_running: ContextVar[bool] = ContextVar(
    "deep_research_running", default=False
)


def get_chat_context() -> Optional[ChatRequestContext]:
    return current_chat_context.get()


def get_progress_callback() -> Optional[ProgressCallback]:
    return progress_callback.get()
