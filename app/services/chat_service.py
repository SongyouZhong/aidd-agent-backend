"""Chat service — manual ReAct loop with token-level SSE streaming.

Implements the pattern used by ChatGPT / Claude / Gemini:
    POST /chat  →  text/event-stream (SSE)

The core ``stream_chat()`` async generator orchestrates:
    1. Load history  (Redis hot cache / SeaweedFS cold)
    2. Save user message
    3. Manual ReAct loop:
       - Stream LLM response token-by-token → yield content_delta events
       - If tool calls → execute, yield tool events, loop back
       - If no tool calls → done
    4. Save assistant message + traces to S3
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.agent.llm_provider import StreamChunk, get_default_provider, reset_failed_models
from app.agent.prompt_renderer import assistant_prefill, render_system_prompt
from app.core.config import settings
from app.storage.manager import append_message, load_messages
from app.storage.s3 import s3_storage, trace_key
from app.tools import default_registry, tool_search

logger = logging.getLogger(__name__)

# Guard against infinite tool loops.
MAX_TOOL_ROUNDS = 10


async def stream_chat(
    session_id: str,
    user_content: str,
    user_id: str,
) -> AsyncGenerator[str, None]:
    """Execute agent conversation and yield SSE events.

    This implements a manual ReAct loop with token-level streaming —
    the same approach production systems (ChatGPT, Claude) use.
    """
    # ----- 0. Reset failed models for this new dialogue round -----
    reset_failed_models()

    # ----- 1. Load history -----
    history = await load_messages(session_id)

    # ----- 2. Save user message -----
    user_msg_id = str(uuid.uuid4())
    await append_message(session_id, {
        "id": user_msg_id,
        "role": "user",
        "content": user_content,
        "ts": _now_iso(),
    })

    # ----- 3. Build provider & context -----
    provider = get_default_provider()
    messages: list[BaseMessage] = _history_to_langchain(history)
    messages.append(HumanMessage(content=user_content))

    hot_loaded: set[str] = set()
    active_tools = default_registry.bind_active(hot_loaded=hot_loaded)
    active_names = ["tool_search"] + [t.name for t in active_tools]

    assistant_msg_id = str(uuid.uuid4())
    trace_steps: list[dict[str, Any]] = []

    yield _sse({"event": "message_start", "data": {"message_id": assistant_msg_id}})

    full_text = ""
    total_rounds = 0

    # ----- 4. ReAct loop -----
    while total_rounds < MAX_TOOL_ROUNDS:
        total_rounds += 1

        # Render system prompt fresh each turn (dynamic tool list, time)
        system = SystemMessage(
            content=render_system_prompt(
                active_tools=active_names,
                hot_loaded=hot_loaded,
            )
        )

        # Rebuild tools list (may have changed via hot-loading)
        active_tools = default_registry.bind_active(hot_loaded=hot_loaded)
        tools_for_llm = active_tools  # LangChain StructuredTool list

        llm_messages = [system, *_strip_system(messages)]

        # --- Stream LLM response token-by-token ---
        round_text = ""
        tool_calls: list[StreamChunk] = []
        round_start = time.monotonic()

        try:
            async for chunk in provider.stream(llm_messages, tools=tools_for_llm):
                if chunk.type == "thinking":
                    yield _sse({"event": "thinking_delta", "data": {"delta": chunk.content}})
                elif chunk.type == "text":
                    round_text += chunk.content
                    full_text += chunk.content
                    yield _sse({"event": "content_delta", "data": {"delta": chunk.content}})

                elif chunk.type == "tool_call":
                    tool_calls.append(chunk)
        except Exception as exc:
            logger.exception("LLM stream error")
            yield _sse({"event": "error", "data": {"code": "llm_error", "message": str(exc)}})
            break

        round_ms = int((time.monotonic() - round_start) * 1000)

        # Record LLM trace step
        trace_steps.append({
            "step_number": len(trace_steps) + 1,
            "step_type": "think",
            "latency_ms": round_ms,
            "created_at": _now_iso(),
        })

        # If no tool calls, we're done
        if not tool_calls:
            # Append AI message to the message list
            messages.append(AIMessage(content=round_text))
            break

        # Append AI message with tool calls
        ai_msg = AIMessage(
            content=round_text,
            tool_calls=[
                {"id": tc.tool_call_id, "name": tc.tool_name, "args": tc.tool_args}
                for tc in tool_calls
            ],
        )
        messages.append(ai_msg)

        # --- Execute tools ---
        for tc in tool_calls:
            yield _sse({
                "event": "tool_use_start",
                "data": {
                    "tool_name": tc.tool_name,
                    "tool_call_id": tc.tool_call_id,
                    "args": tc.tool_args,
                },
            })

            tool_start = time.monotonic()
            result = await _execute_tool(tc.tool_name, tc.tool_args)
            tool_ms = int((time.monotonic() - tool_start) * 1000)

            # Auto-mount tools surfaced by tool_search
            if tc.tool_name == "tool_search":
                try:
                    payload = json.loads(result)
                    for m in payload.get("matches", []):
                        if m.get("name"):
                            hot_loaded.add(m["name"])
                    active_names = ["tool_search"] + [
                        t.name for t in default_registry.bind_active(hot_loaded=hot_loaded)
                    ] 
                except Exception:
                    pass

            result_summary = str(result)[:200] + ("..." if len(str(result)) > 200 else "")

            yield _sse({
                "event": "tool_use_end",
                "data": {
                    "tool_call_id": tc.tool_call_id,
                    "result_summary": result_summary,
                },
            })

            messages.append(
                ToolMessage(content=str(result), name=tc.tool_name, tool_call_id=tc.tool_call_id)
            )

            # Record tool trace step
            trace_steps.append({
                "step_number": len(trace_steps) + 1,
                "step_type": "act",
                "tool_name": tc.tool_name,
                "tool_args": tc.tool_args,
                "tool_result_summary": result_summary,
                "latency_ms": tool_ms,
                "created_at": _now_iso(),
            })

        # Loop back to LLM with tool results
        round_text = ""

    # ----- 5. Send message_end -----
    yield _sse({
        "event": "message_end",
        "data": {
            "message_id": assistant_msg_id,
            "usage": {"output_tokens": _estimate_tokens(full_text)},
        },
    })
    yield "data: [DONE]\n\n"

    # ----- 6. Save assistant message -----
    await append_message(session_id, {
        "id": assistant_msg_id,
        "role": "assistant",
        "content": full_text,
        "ts": _now_iso(),
    })

    # ----- 7. Save traces to S3 -----
    if trace_steps:
        try:
            key = trace_key(session_id, assistant_msg_id)
            traces_data = "\n".join(json.dumps(t, ensure_ascii=False) for t in trace_steps)
            await s3_storage.put_object(key, traces_data, content_type="application/x-ndjson")
        except Exception:
            logger.exception("Failed to save traces to S3")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sse(payload: dict[str, Any]) -> str:
    """Format a single SSE data line."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _strip_system(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if not isinstance(m, SystemMessage)]


def _history_to_langchain(history: list[dict[str, Any]]) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


async def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Look up and invoke a registered tool by name."""
    if name == "tool_search":
        return tool_search.invoke(args)
    impl = default_registry.get(name)
    if impl is None:
        return f"[error] tool '{name}' is not loaded"
    if getattr(impl, "coroutine", None) is not None:
        result = await impl.ainvoke(args)
    else:
        result = impl.invoke(args)
    return str(result)
