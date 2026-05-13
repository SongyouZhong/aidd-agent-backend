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

import asyncio
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

from app.agent.context_manager import (
    CompactTrackingState,
    apply_compaction,
    maybe_compact,
)
from app.agent.llm_provider import StreamChunk, get_default_provider, reset_failed_models
from app.agent.prompt_renderer import render_system_prompt
from app.core.config import settings
from app.services.chat_context import (
    ChatRequestContext,
    current_chat_context,
    progress_callback,
)
from app.services import task_registry
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
    plan_mode: bool = False,
    file_ids: list[str] | None = None,
    project_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Execute agent conversation and yield SSE events.

    This implements a manual ReAct loop with token-level streaming —
    the same approach production systems (ChatGPT, Claude) use.
    """
    # ----- 0. Reset failed models for this new dialogue round -----
    reset_failed_models()

    # ----- 0.5 Set per-request context for tools (e.g. run_target_discovery) -----
    current_chat_context.set(
        ChatRequestContext(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
        )
    )
    # Bridge between background tools and the SSE generator.
    progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def _push_progress(event_type: str, payload: dict) -> None:
        await progress_queue.put({"event": event_type, "data": payload})

    progress_callback.set(_push_progress)

    # Track tool-produced session files so we can attach them to the
    # final assistant message (e.g. deep-research report downloads).
    produced_file_ids: list[str] = []

    # Auto-Compaction tracking (was previously held in agent_node state).
    compact_tracking = CompactTrackingState()

    # ----- 1. Load history -----
    history = await load_messages(session_id)

    # ----- 1.5 Load file context if file_ids provided -----
    file_context = ""
    if file_ids:
        file_context = await _load_file_context(session_id, file_ids)

    # ----- 2. Save user message -----
    user_msg_id = str(uuid.uuid4())
    await append_message(session_id, {
        "id": user_msg_id,
        "role": "user",
        "content": user_content,
        "ts": _now_iso(),
        "file_ids": file_ids or [],
    })

    # ----- 3. Build provider & context -----
    provider = get_default_provider()
    messages: list[BaseMessage] = _history_to_langchain(history)

    # Inject file context into the user message if present
    enriched_content = user_content
    if file_context:
        enriched_content = f"{user_content}\n\n--- 附件内容 ---\n{file_context}"
    messages.append(HumanMessage(content=enriched_content))

    hot_loaded: set[str] = set()
    active_tools = default_registry.bind_active(hot_loaded=hot_loaded)
    active_names = ["tool_search"] + [t.name for t in active_tools]

    assistant_msg_id = str(uuid.uuid4())
    trace_steps: list[dict[str, Any]] = []

    full_text = ""
    total_rounds = 0

    try:
        yield _sse({"event": "message_start", "data": {"message_id": assistant_msg_id}})

        # ----- 4. ReAct loop -----
        while total_rounds < MAX_TOOL_ROUNDS:
            total_rounds += 1

            # Auto-Compaction: shrink history before sending to the LLM if we're
            # over threshold (design doc §9).
            async def _summarizer(msgs: list[BaseMessage]) -> str:
                resp = await provider.generate(messages=msgs, tools=None)
                return resp.text

            try:
                _current_model = (
                    getattr(provider, "model", None)
                    or (provider.primaries[0].model if getattr(provider, "primaries", None) else None)
                    or settings.GEMINI_MODELS.split(",")[0].strip()
                )
                compact_result = await maybe_compact(
                    messages,
                    model=_current_model,
                    tracking=compact_tracking,
                    summarizer=_summarizer,
                )
                if compact_result is not None:
                    messages = apply_compaction(list(messages), compact_result)
            except Exception:
                logger.exception("Auto-compaction failed; continuing without")

            # Render system prompt fresh each turn (dynamic tool list, time)
            system = SystemMessage(
                content=render_system_prompt(
                    active_tools=active_names,
                    hot_loaded=hot_loaded,
                    system_status="plan_mode" if plan_mode else "ready",
                )
            )

            # Rebuild tools list (may have changed via hot-loading)
            active_tools = default_registry.bind_active(hot_loaded=hot_loaded)
            tools_for_llm = [tool_search] + active_tools  # always expose tool_search schema

            llm_messages = [system, *_strip_system(messages)]

            # --- Stream LLM response token-by-token ---
            round_text = ""
            round_thinking = ""
            tool_calls: list[StreamChunk] = []
            round_start = time.monotonic()

            try:
                async for chunk in provider.stream(llm_messages, tools=tools_for_llm):
                    if chunk.type == "thinking":
                        round_thinking += chunk.content
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
                ai_content = (
                    f"<thought>\n{round_thinking}\n</thought>\n\n{round_text}"
                    if round_thinking else round_text
                )
                messages.append(AIMessage(content=ai_content))
                break

            # Append AI message with tool calls
            ai_content = (
                f"<thought>\n{round_thinking}\n</thought>\n\n{round_text}"
                if round_thinking else round_text
            )
            ai_msg = AIMessage(
                content=ai_content,
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
                # Long-running tools (currently: run_target_discovery) push
                # ``research_progress`` events through the progress queue.
                # Run them in a background task so we can drain that queue
                # and forward events to the SSE stream while the tool works.
                tool_task = asyncio.create_task(
                    _execute_tool(tc.tool_name, tc.tool_args)
                )
                try:
                    last_heartbeat = time.monotonic()
                    while not tool_task.done():
                        try:
                            ev = await asyncio.wait_for(
                                progress_queue.get(), timeout=0.5
                            )
                            last_heartbeat = time.monotonic()
                            yield _sse(ev)
                        except asyncio.TimeoutError:
                            now = time.monotonic()
                            if now - last_heartbeat >= 15:
                                # Keep long-running tool streams alive through
                                # Cloudflare/HTTP2 while a graph node is working
                                # and has not emitted a progress event yet.
                                yield ": keep-alive\n\n"
                                last_heartbeat = now
                            continue
                    # Drain any final queued events.
                    while not progress_queue.empty():
                        yield _sse(progress_queue.get_nowait())
                    result = tool_task.result()
                except asyncio.CancelledError:
                    tool_task.cancel()
                    raise
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

                # When run_target_discovery returns 'accepted', associate the
                # tool_call_id with the background task so resume_after_task can
                # later inject the correct ToolMessage into history.
                if tc.tool_name == "run_target_discovery":
                    try:
                        payload = json.loads(result)
                        if payload.get("status") == "accepted" and payload.get("task_id"):
                            await task_registry.set_tool_call_id(
                                payload["task_id"], tc.tool_call_id
                            )
                    except Exception:
                        logger.exception("Failed to register tool_call_id for background task")

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
            round_thinking = ""

        # ----- 4.5 Emit citation events -----
        # Extract URLs/references from tool results for frontend citation rendering
        citations = _extract_citations(full_text, trace_steps)
        for i, cite in enumerate(citations):
            yield _sse({
                "event": "citation",
                "data": {
                    "index": i + 1,
                    "url": cite.get("url", ""),
                    "title": cite.get("title", ""),
                },
            })

        # ----- 5. Auto-generate session title on first message -----
        if not history:
            try:
                new_title = await _generate_session_title(
                    user_content, full_text, session_id, user_id, project_id
                )
                if new_title:
                    yield _sse({
                        "event": "session_title_updated",
                        "data": {"title": new_title},
                    })
            except Exception:
                logger.exception("Auto-title generation failed; session keeps default title")

        # ----- 6. Send message_end -----
        yield _sse({
            "event": "message_end",
            "data": {
                "message_id": assistant_msg_id,
                "usage": {"output_tokens": _estimate_tokens(full_text)},
            },
        })
        yield "data: [DONE]\n\n"
    finally:
        # Phase 0: protect persistence from client disconnection.
        # asyncio.shield keeps _finalize running even if this generator
        # is cancelled (CancelledError thrown at a yield point).
        try:
            await asyncio.shield(
                _finalize(
                    session_id, assistant_msg_id, full_text,
                    produced_file_ids, trace_steps,
                )
            )
        except asyncio.CancelledError:
            pass



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


def _extract_citations(
    full_text: str, trace_steps: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Extract unique URLs from tool results and response text as citations.

    Returns a list of dicts with 'url' and 'title' keys.
    """
    import re

    url_pattern = re.compile(r"https?://[^\s\)\]\}\"'<>]+")
    seen: set[str] = set()
    citations: list[dict[str, str]] = []

    # Scan tool result summaries for URLs (e.g., PubMed links)
    for step in trace_steps:
        summary = step.get("tool_result_summary", "") or ""
        for url in url_pattern.findall(summary):
            url = url.rstrip(".,;:")
            if url not in seen:
                seen.add(url)
                # Derive a short title from the URL
                title = _url_to_title(url)
                citations.append({"url": url, "title": title})

    # Also scan the full response text
    for url in url_pattern.findall(full_text):
        url = url.rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            citations.append({"url": url, "title": _url_to_title(url)})

    return citations


def _url_to_title(url: str) -> str:
    """Best-effort short title from a URL."""
    if "pubmed.ncbi.nlm.nih.gov" in url:
        pmid = url.rstrip("/").split("/")[-1]
        return f"PubMed: {pmid}"
    if "doi.org/" in url:
        doi = url.split("doi.org/", 1)[-1]
        return f"DOI: {doi}"
    if "uniprot.org/" in url:
        uid = url.rstrip("/").split("/")[-1]
        return f"UniProt: {uid}"
    # Fallback: use domain
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    return domain or url[:60]


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


async def _load_file_context(session_id: str, file_ids: list[str]) -> str:
    """Load file contents from S3 for the given file IDs.

    Returns a combined string of file contents suitable for injecting
    into the user message as context.
    """
    from sqlalchemy import select
    from app.db.engine import AsyncSessionLocal
    from app.models.session_file import SessionFile

    parts: list[str] = []

    async with AsyncSessionLocal() as db:
        for fid in file_ids:
            try:
                result = await db.execute(
                    select(SessionFile).where(
                        SessionFile.id == uuid.UUID(fid),
                        SessionFile.session_id == uuid.UUID(session_id),
                    )
                )
                record = result.scalar_one_or_none()
                if record is None:
                    parts.append(f"[File {fid}]: (file not found)")
                    continue

                raw = await s3_storage.get_object(record.s3_key)
                if raw:
                    try:
                        text = raw.decode("utf-8")
                        # Truncate very long files
                        parts.append(f"[File: {record.original_filename}]:\n{text[:20000]}")
                    except UnicodeDecodeError:
                        parts.append(
                            f"[File: {record.original_filename}]: "
                            f"(binary file, {len(raw)} bytes, type: {record.mime_type})"
                        )
                else:
                    parts.append(f"[File: {record.original_filename}]: (S3 object missing)")
            except Exception:
                logger.warning("Failed to load file context for %s", fid)
                parts.append(f"[File {fid}]: (load error)")
    return "\n\n".join(parts)


async def _generate_session_title(
    user_content: str,
    assistant_content: str,
    session_id: str,
    user_id: str,
    project_id: str | None,
) -> str | None:
    """Generate a short session title using the local Qwen model.

    Called once after the first message in a session.  Uses the local vLLM
    instance so there is zero cloud-API cost.  Returns the new title string,
    or None if generation fails or yields an empty result.
    """
    from app.agent.llm_provider import QwenProvider
    from app.db.engine import AsyncSessionLocal
    from app.services import session_service

    # Truncate inputs to keep the prompt tiny.
    user_snippet = user_content[:200]
    ai_snippet = assistant_content[:200]

    prompt = (
        "请根据以下对话内容，生成一个不超过15个字的简短标题。"
        "只输出标题本身，不要加引号、序号或任何额外内容。\n\n"
        f"用户: {user_snippet}\n"
        f"助手: {ai_snippet}"
    )

    try:
        title_provider = QwenProvider()
        resp = await title_provider.generate(
            messages=[HumanMessage(content=prompt)],
            tools=None,
            max_tokens=30,
        )
        title = resp.text.strip().strip('"\'""''')
        if not title or len(title) > 50:
            return None
    except Exception:
        logger.warning("Local Qwen title generation failed", exc_info=True)
        return None

    # Persist to database.
    try:
        async with AsyncSessionLocal() as db:
            await session_service.rename_session(
                db,
                uuid.UUID(session_id),
                uuid.UUID(user_id),
                title,
                project_id=uuid.UUID(project_id) if project_id else None,
            )
        logger.info("Auto-titled session %s → %r", session_id, title)
        return title
    except Exception:
        logger.warning("Failed to persist auto-generated title", exc_info=True)
        return None


async def _finalize(
    session_id: str,
    assistant_msg_id: str,
    full_text: str,
    produced_file_ids: list[str],
    trace_steps: list[dict[str, Any]],
) -> None:
    """Persist the assistant message and trace to storage.

    Called inside ``asyncio.shield`` from ``stream_chat``'s ``finally``
    block so it completes even when the SSE stream is cancelled
    (e.g. client refresh mid-stream).
    """
    await append_message(session_id, {
        "id": assistant_msg_id,
        "role": "assistant",
        "content": full_text,
        "ts": _now_iso(),
        "file_ids": produced_file_ids,
    })
    if trace_steps:
        try:
            key = trace_key(session_id, assistant_msg_id)
            traces_data = "\n".join(json.dumps(t, ensure_ascii=False) for t in trace_steps)
            await s3_storage.put_object(key, traces_data, content_type="application/x-ndjson")
        except Exception:
            logger.exception("Failed to save traces to S3")


async def resume_after_task(session_id: str, task_id: str) -> None:
    """Generate and persist an assistant summary after a background task completes.

    Called by the background task coroutine (not via any HTTP connection).
    Loads persisted history, calls LLM for a brief completion notification,
    persists the result, and broadcasts ``message_appended`` so connected
    clients know to refresh the message list.
    """
    from app.agent.prompt_renderer import render_system_prompt

    task = await task_registry.get(task_id)
    if not task:
        logger.warning("resume_after_task: task %s not found", task_id)
        return

    file_ids: list[str] = task.result.get("file_ids", []) if task.result else []

    # Build a minimal prompt so the LLM can generate a short user-facing notice.
    provider = get_default_provider()
    history = await load_messages(session_id)
    lc_msgs = _history_to_langchain(history)

    system = SystemMessage(content=render_system_prompt(active_tools=[], hot_loaded=set()))
    notify_prompt = HumanMessage(content=(
        f"[Internal system notification: The background target discovery pipeline "
        f"for '{task.target}' has just completed. "
        f"The full Markdown report is saved as a session file. "
        f"Please write a brief 1-2 sentence message for the user telling them "
        f"the analysis is complete and the full report is ready in the side panel. "
        f"Do not include any raw research data or long lists.]"
    ))

    llm_messages = [system, *_strip_system(lc_msgs), notify_prompt]

    try:
        resp = await provider.generate(llm_messages, tools=None)
        summary = resp.text or (
            f"The target discovery analysis for **{task.target}** is complete. "
            f"The full report is now available in the side panel."
        )
    except Exception:
        logger.exception("resume_after_task: LLM call failed; using fallback message")
        summary = (
            f"The target discovery analysis for **{task.target}** is complete. "
            f"The full report is now available in the side panel."
        )

    msg_id = str(uuid.uuid4())
    await append_message(session_id, {
        "id": msg_id,
        "role": "assistant",
        "content": summary,
        "ts": _now_iso(),
        "file_ids": file_ids,
    })

    await task_registry.publish_event(session_id, {
        "type": "message_appended",
        "session_id": session_id,
        "message_id": msg_id,
    })
