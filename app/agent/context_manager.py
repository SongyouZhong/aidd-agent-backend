"""Auto-Compaction + Circuit Breaker (design doc §9).

This module is intentionally LLM-agnostic: pass in a callable
``summarizer(messages) -> str`` so unit tests stay offline.

Public surface:
    * ``count_tokens(text)`` / ``count_tokens_messages(messages)``
    * ``get_auto_compact_threshold(model)``
    * ``CompactTrackingState`` (per-session, in-memory)
    * ``maybe_compact(state, *, model, summarizer, session_memory)``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.agent.prompts.templates import COMPACT_PROMPT
from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Token counting ---------------------------------------------------

def count_tokens(text: str) -> int:
    """Rough char/3 estimate (consistent with app.tools.preprocess)."""
    return max(1, len(text or "") // 3)


def count_tokens_messages(messages: Sequence[BaseMessage]) -> int:
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += count_tokens(content)
        # Tool calls + names contribute to the wire payload too.
        for tc in getattr(m, "tool_calls", None) or []:
            total += count_tokens(str(tc))
    return total


def get_effective_context_window(model: str) -> int:
    is_local = (model == settings.QWEN_MODEL)
    window = settings.LOCAL_CONTEXT_WINDOW if is_local else settings.EXTERNAL_CONTEXT_WINDOW
    return window - settings.MAX_OUTPUT_TOKENS_FOR_SUMMARY


def get_auto_compact_threshold(model: str) -> int:
    window = get_effective_context_window(model)
    return int(window * settings.AUTOCOMPACT_THRESHOLD_PERCENT)


# --- Per-session tracking --------------------------------------------

@dataclass
class CompactTrackingState:
    compacted: bool = False
    turn_counter: int = 0
    consecutive_failures: int = 0
    last_summarized_message_id: str | None = None


@dataclass
class CompactionResult:
    summary_messages: list[BaseMessage]
    messages_to_keep: list[BaseMessage]
    archived: list[BaseMessage] = field(default_factory=list)
    method: str = "unknown"


# --- Decision -------------------------------------------------------

def should_auto_compact(
    messages: Sequence[BaseMessage],
    *,
    model: str,
    tracking: CompactTrackingState,
) -> bool:
    if settings.DISABLE_AUTO_COMPACT:
        return False
    if tracking.consecutive_failures >= settings.MAX_CONSECUTIVE_COMPACT_FAILURES:
        return False
    return count_tokens_messages(messages) >= get_auto_compact_threshold(model)


# --- Compaction strategies -------------------------------------------

def _calculate_keep_index(
    messages: Sequence[BaseMessage],
    *,
    min_tokens: int,
    min_text_messages: int,
    max_tokens: int,
) -> int:
    """Return the start index of messages to KEEP after compaction.

    Walks from the tail collecting tokens until either ``min_tokens`` and
    ``min_text_messages`` are both satisfied, then keeps going until
    ``max_tokens`` would be exceeded.
    """
    total = 0
    text_msgs = 0
    keep_from = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        size = count_tokens(m.content if isinstance(m.content, str) else str(m.content))
        if total + size > max_tokens and total >= min_tokens and text_msgs >= min_text_messages:
            break
        total += size
        if isinstance(m, (HumanMessage, AIMessage)):
            text_msgs += 1
        keep_from = i
        if total >= min_tokens and text_msgs >= min_text_messages and total >= max_tokens:
            break
    return keep_from


def try_session_memory_compaction(
    messages: Sequence[BaseMessage],
    *,
    session_memory: str | None,
) -> CompactionResult | None:
    """Level 1 compaction — uses pre-extracted Session Memory, no LLM call."""
    if not session_memory or not session_memory.strip():
        return None
    keep_from = _calculate_keep_index(
        messages,
        min_tokens=10_000,
        min_text_messages=5,
        max_tokens=40_000,
    )
    archived = list(messages[:keep_from])
    kept = list(messages[keep_from:])
    summary_msg = SystemMessage(
        content=(
            "[Session Memory — 前序对话摘要]\n"
            f"{session_memory}\n\n"
            "以下是近期保留的原始对话，请基于摘要和近期上下文继续工作。"
        )
    )
    return CompactionResult(
        summary_messages=[summary_msg],
        messages_to_keep=kept,
        archived=archived,
        method="session_memory",
    )


async def llm_compaction(
    messages: Sequence[BaseMessage],
    *,
    summarizer: Callable[[list[BaseMessage]], Awaitable[str]],
) -> CompactionResult:
    """Level 2 compaction — full LLM summarization (design doc §9.6)."""
    summary_text = await summarizer(
        [
            SystemMessage(content=COMPACT_PROMPT),
            *messages,
            HumanMessage(content="请基于以上对话生成结构化摘要。"),
        ]
    )
    return CompactionResult(
        summary_messages=[
            SystemMessage(content=f"[Auto-Compacted Summary]\n{summary_text}")
        ],
        messages_to_keep=[],
        archived=list(messages),
        method="llm_summary",
    )


async def maybe_compact(
    messages: Sequence[BaseMessage],
    *,
    model: str,
    tracking: CompactTrackingState,
    summarizer: Callable[[list[BaseMessage]], Awaitable[str]] | None = None,
    session_memory: str | None = None,
) -> CompactionResult | None:
    """Run the full Auto-Compaction flow if the threshold is reached.

    Returns ``None`` if nothing happened (below threshold, disabled, or
    circuit-breaker tripped).
    """
    if not should_auto_compact(messages, model=model, tracking=tracking):
        return None

    # Level 1
    result = try_session_memory_compaction(messages, session_memory=session_memory)
    if result is not None:
        tracking.compacted = True
        tracking.consecutive_failures = 0
        tracking.turn_counter = 0
        return result

    # Level 2
    if summarizer is None:
        tracking.consecutive_failures += 1
        logger.warning(
            "Compaction needed but no summarizer available (failures=%d)",
            tracking.consecutive_failures,
        )
        return None
    try:
        result = await llm_compaction(messages, summarizer=summarizer)
    except Exception:
        tracking.consecutive_failures += 1
        logger.exception(
            "LLM compaction failed (failures=%d)", tracking.consecutive_failures
        )
        return None

    tracking.compacted = True
    tracking.consecutive_failures = 0
    tracking.turn_counter = 0
    return result


def apply_compaction(
    messages: list[BaseMessage], result: CompactionResult
) -> list[BaseMessage]:
    """Replace ``messages`` content with summary + retained tail."""
    return [*result.summary_messages, *result.messages_to_keep]


# Re-export for convenience.
__all__ = [
    "CompactTrackingState",
    "CompactionResult",
    "apply_compaction",
    "count_tokens",
    "count_tokens_messages",
    "get_auto_compact_threshold",
    "get_effective_context_window",
    "llm_compaction",
    "maybe_compact",
    "should_auto_compact",
    "try_session_memory_compaction",
]


# Note: ``ToolMessage`` is intentionally imported but not specially treated;
# it counts toward the token budget like any other message.
_ = ToolMessage  # silence unused-import linters
