"""Tool-layer preprocessing utilities.

Implements the mandatory three-step pipeline from the backend design doc §7.2.1:

    1. Hard pruning  (handled by Pydantic schemas in ``app.tools.schemas``)
    2. Semantic re-ranking / chunking helpers (here)
    3. Map-Reduce summarisation (``app.tools.mapreduce``)

Plus a hard guard: NO tool may return more than ``MAX_TOOL_TOKENS`` tokens
(AC §2.2 — "any single tool payload must never exceed 40000 tokens").
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

# A safe approximation: roughly 4 chars per token for English, ~1.5 for CJK.
# We split the difference at 3 so guarded payloads err on the safe side.
_CHARS_PER_TOKEN = 3

# Hard ceiling enforced by ``cap_tokens`` and ``@guarded_tool``.
MAX_TOOL_TOKENS = 40_000
TRUNCATION_NOTICE = "\n\n…[truncated by preprocessing pipeline; raw output stored as sidechain]"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def cap_tokens(text: str, max_tokens: int = MAX_TOOL_TOKENS) -> str:
    """Truncate ``text`` so its estimated token count does not exceed the cap.

    Truncation happens at a paragraph boundary if possible to preserve
    readability for the downstream LLM. The truncation notice itself is
    budgeted for, so the final string is guaranteed under ``max_tokens``.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    notice_tokens = estimate_tokens(TRUNCATION_NOTICE)
    budget = max(1, max_tokens - notice_tokens)
    max_chars = budget * _CHARS_PER_TOKEN
    cut = text[:max_chars]
    boundary = max(cut.rfind("\n\n"), cut.rfind(". "))
    if boundary > max_chars * 0.5:
        cut = cut[:boundary]
    return cut.rstrip() + TRUNCATION_NOTICE


def chunk_text(text: str, target_tokens: int = 500) -> list[str]:
    """Split ``text`` into chunks of approximately ``target_tokens`` tokens.

    Used by the (future) embedding-based re-ranker. Splits on paragraph
    boundaries first, falling back to a hard char split.
    """
    target_chars = target_tokens * _CHARS_PER_TOKEN
    if len(text) <= target_chars:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        if buf_len + len(para) > target_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2

    if buf:
        chunks.append("\n\n".join(buf))

    # Fallback hard-split for any chunk that's still oversized.
    out: list[str] = []
    for c in chunks:
        if len(c) <= target_chars * 1.5:
            out.append(c)
        else:
            out.extend(c[i : i + target_chars] for i in range(0, len(c), target_chars))
    return out


P = ParamSpec("P")
R = TypeVar("R", bound=str)


def guarded_tool(
    max_tokens: int = MAX_TOOL_TOKENS,
) -> Callable[[Callable[P, R | Awaitable[R]]], Callable[P, R | Awaitable[R]]]:
    """Decorator that enforces ``cap_tokens`` on a tool's string return value.

    Works transparently with both sync and async functions.
    """

    def deco(fn: Callable[P, R | Awaitable[R]]) -> Callable[P, R | Awaitable[R]]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                out = await fn(*args, **kwargs)
                return cap_tokens(out, max_tokens) if isinstance(out, str) else out

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def swrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            out = fn(*args, **kwargs)
            return cap_tokens(out, max_tokens) if isinstance(out, str) else out

        return swrapper  # type: ignore[return-value]

    return deco
