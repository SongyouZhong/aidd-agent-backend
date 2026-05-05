"""LLM provider — wraps Gemini (google-genai) + offline fake for tests.

Public contract:
    ``await provider.generate(messages, tools=...) -> AIResponse``

``AIResponse`` carries the answer text, optional grounding metadata, and
the list of tool calls the model wants to perform. Both real and fake
providers obey the same contract so the LangGraph node is identical.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRequest:
    name: str
    args: dict[str, Any]
    id: str


@dataclass
class AIResponse:
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    grounding_metadata: Any = None
    raw: Any = None


# --- Fake provider (offline-friendly) ----------------------------------

class FakeLLMProvider:
    """Deterministic provider for unit tests / no-API-key dev.

    Behaviour is driven by an injected ``script`` of pre-baked
    ``AIResponse`` instances, consumed in order.
    """

    def __init__(self, script: list[AIResponse]) -> None:
        self._script = list(script)

    async def generate(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AIResponse:
        if not self._script:
            return AIResponse(text="<answer>(no scripted response)</answer>")
        return self._script.pop(0)


# --- Gemini provider ---------------------------------------------------

class GeminiProvider:
    """Real Gemini provider with optional Google Search grounding."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        enable_grounding: bool = True,
    ) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.enable_grounding = enable_grounding
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        from google import genai  # type: ignore

        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Normalise LangChain message content to a plain string.

        LangGraph Studio (and OpenAI-compatible clients) may send content as
        a list of content blocks, e.g. [{"type": "text", "text": "..."}].
        Gemini requires a plain string for the ``text`` part field.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _to_gemini_contents(messages: list[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
        system_text = ""
        contents: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_text += GeminiProvider._extract_text(m.content) + "\n"
            elif isinstance(m, HumanMessage):
                contents.append({"role": "user", "parts": [{"text": GeminiProvider._extract_text(m.content)}]})
            elif isinstance(m, AIMessage):
                contents.append({"role": "model", "parts": [{"text": GeminiProvider._extract_text(m.content)}]})
            elif isinstance(m, ToolMessage):
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"[tool:{m.name}] {GeminiProvider._extract_text(m.content)}"}],
                })
        return system_text, contents

    @staticmethod
    def _langchain_tools_to_declarations(tools: list[Any]) -> list[dict[str, Any]]:
        """Convert LangChain StructuredTools into Gemini FunctionDeclarations."""
        declarations = []
        for t in (tools or []):
            schema = {}
            if hasattr(t, "args_schema") and t.args_schema is not None:
                raw = t.args_schema.model_json_schema()
                # Build a clean JSON-Schema-style 'parameters' object.
                props = {}
                for pname, pdef in raw.get("properties", {}).items():
                    prop: dict[str, Any] = {}
                    json_type = pdef.get("type", "string")
                    type_map = {
                        "string": "STRING", "integer": "INTEGER",
                        "number": "NUMBER", "boolean": "BOOLEAN",
                        "array": "ARRAY", "object": "OBJECT",
                    }
                    prop["type"] = type_map.get(json_type, "STRING")
                    if "description" in pdef:
                        prop["description"] = pdef["description"]
                    props[pname] = prop
                schema = {
                    "type": "OBJECT",
                    "properties": props,
                    "required": raw.get("required", []),
                }
            declarations.append({
                "name": t.name,
                "description": (t.description or "")[:500],
                "parameters": schema if schema else None,
            })
        return declarations

    async def generate(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AIResponse:
        system_text, contents = self._to_gemini_contents(messages)
        from google.genai import types  # type: ignore

        config_kwargs: dict[str, Any] = {}
        if system_text:
            config_kwargs["system_instruction"] = system_text

        # Build tool list: Google Search grounding and Function Calling
        # are MUTUALLY EXCLUSIVE in the Gemini API. When we have tools
        # to register as function declarations, we must use Function
        # Calling only; otherwise we can use Google Search grounding.
        gemini_tools: list[Any] = []
        if tools:
            func_decls = self._langchain_tools_to_declarations(tools)
            if func_decls:
                gemini_tools.append(types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**d) for d in func_decls
                    ]
                ))
        if not gemini_tools and self.enable_grounding:
            # Only use Google Search when no function declarations are present
            gemini_tools.append(types.Tool(google_search=types.GoogleSearch()))
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        resp = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self.model,
            contents=contents,
            config=config,
        )

        # Extract text (may be empty when model returns function calls)
        text = ""
        tool_call_requests: list[ToolCallRequest] = []
        grounding_metadata = None
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            candidate = candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            content = getattr(candidate, "content", None)
            if content:
                for part in getattr(content, "parts", []) or []:
                    if hasattr(part, "text") and part.text:
                        text += part.text
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_call_requests.append(ToolCallRequest(
                            name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                            id=f"call-{fc.name}-{len(tool_call_requests)}",
                        ))

        # Only fall back to resp.text when there are no tool calls.
        # Accessing resp.text when parts contain function_call triggers a
        # spurious UserWarning from the Google GenAI SDK.
        if not text and not tool_call_requests:
            text = getattr(resp, "text", "") or ""

        return AIResponse(
            text=text,
            tool_calls=tool_call_requests,
            grounding_metadata=grounding_metadata,
            raw=resp,
        )


# --- Qwen provider (local vLLM, OpenAI-compatible) -------------------

class QwenProvider:
    """Local Qwen3 served by vLLM via OpenAI-compatible API.

    Used as a fallback when Gemini is unavailable (503 / 429).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = base_url or settings.QWEN_BASE_URL
        self.model = model or settings.QWEN_MODEL
        self.api_key = api_key or settings.QWEN_API_KEY
        self.enable_thinking = enable_thinking
        if not self.base_url:
            raise RuntimeError("QWEN_BASE_URL not configured")
        from openai import AsyncOpenAI  # type: ignore

        self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    @staticmethod
    def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            text = GeminiProvider._extract_text(m.content)
            if isinstance(m, SystemMessage):
                out.append({"role": "system", "content": text})
            elif isinstance(m, HumanMessage):
                out.append({"role": "user", "content": text})
            elif isinstance(m, AIMessage):
                msg_dict: dict[str, Any] = {"role": "assistant", "content": text or None}
                tc_list = getattr(m, "tool_calls", None)
                if tc_list:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"], ensure_ascii=False),
                            },
                        }
                        for tc in tc_list
                    ]
                out.append(msg_dict)
            elif isinstance(m, ToolMessage):
                # OpenAI requires tool_call_id; fall back to a synthetic one.
                out.append({
                    "role": "tool",
                    "content": text,
                    "tool_call_id": getattr(m, "tool_call_id", None) or f"call-{m.name}",
                })
        return out

    @staticmethod
    def _langchain_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
        decls: list[dict[str, Any]] = []
        for t in tools or []:
            params: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            if hasattr(t, "args_schema") and t.args_schema is not None:
                raw = t.args_schema.model_json_schema()
                params = {
                    "type": "object",
                    "properties": raw.get("properties", {}),
                    "required": raw.get("required", []),
                }
            decls.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "")[:500],
                    "parameters": params,
                },
            })
        return decls

    async def generate(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AIResponse:
        oa_messages = self._to_openai_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oa_messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 4096,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        }
        if tools:
            oa_tools = self._langchain_tools_to_openai(tools)
            if oa_tools:
                kwargs["tools"] = oa_tools

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""

        tool_call_requests: list[ToolCallRequest] = []
        for tc in (msg.tool_calls or []) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (TypeError, ValueError):
                args = {}
            tool_call_requests.append(
                ToolCallRequest(
                    name=tc.function.name,
                    args=args,
                    id=tc.id or f"call-{tc.function.name}-{uuid.uuid4().hex[:8]}",
                )
            )

        return AIResponse(
            text=text,
            tool_calls=tool_call_requests,
            grounding_metadata=None,
            raw=resp,
        )


# --- Fallback wrapper (process-level circuit breaker) ----------------

class FallbackLLMProvider:
    """Primary provider with automatic fallback to a secondary on 503/429.

    A *process-level* circuit breaker is opened when the primary returns
    a retryable infrastructure error (UNAVAILABLE / RESOURCE_EXHAUSTED /
    HTTP 503 / 429). While open, all calls go directly to the secondary.
    After the configured timeout elapses, the next call probes the primary
    again. Other errors (4xx content filtering, auth) propagate as-is.
    """

    # Class-level state shared across instances within one process.
    _circuit_open_until: float = 0.0
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        primary: Any,
        secondary: Any,
        circuit_break_seconds: int | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.circuit_break_seconds = (
            circuit_break_seconds
            if circuit_break_seconds is not None
            else settings.LLM_CIRCUIT_BREAK_SECONDS
        )

    @staticmethod
    def _is_retryable_failure(exc: BaseException) -> bool:
        """Return True for infrastructure failures worth failing over on."""
        # google-genai raises google.genai.errors.{ClientError,ServerError,APIError}
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status in (429, 503):
            return True
        msg = str(exc).upper()
        if "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg:
            return True
        if "503" in msg and ("UNAVAILABLE" in msg or "OVERLOAD" in msg):
            return True
        # Network-level transient errors
        for marker in ("DEADLINE_EXCEEDED", "TIMEOUT", "CONNECTION RESET"):
            if marker in msg:
                return True
        return False

    @classmethod
    def _circuit_is_open(cls) -> bool:
        return cls._circuit_open_until > time.monotonic()

    @classmethod
    async def _open_circuit(cls, seconds: int) -> None:
        async with cls._lock:
            cls._circuit_open_until = time.monotonic() + seconds

    async def generate(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AIResponse:
        # If breaker is open, skip primary entirely.
        if self._circuit_is_open():
            return await self.secondary.generate(messages, tools=tools)

        try:
            return await self.primary.generate(messages, tools=tools)
        except Exception as exc:
            if not self._is_retryable_failure(exc):
                raise
            logger.warning(
                "Primary LLM unavailable (%s: %s); failing over to secondary "
                "for %ds",
                type(exc).__name__,
                exc,
                self.circuit_break_seconds,
            )
            await self._open_circuit(self.circuit_break_seconds)
            return await self.secondary.generate(messages, tools=tools)


# --- Factory -----------------------------------------------------------

def get_default_provider() -> Any:
    """Return the configured provider with optional Qwen fallback.

    Resolution order:
      1. If ``AIDD_FORCE_FAKE_LLM`` is set → ``FakeLLMProvider`` (offline)
      2. If Gemini key + Qwen base URL + fallback enabled → ``FallbackLLMProvider``
      3. If Gemini key only → ``GeminiProvider``
      4. Else → ``FakeLLMProvider``
    """
    if os.environ.get("AIDD_FORCE_FAKE_LLM"):
        return FakeLLMProvider(
            script=[AIResponse(text="<answer>(forced offline mode)</answer>")]
        )

    primary = None
    if settings.GEMINI_API_KEY:
        try:
            primary = GeminiProvider()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to init GeminiProvider: %s", exc)

    secondary = None
    if settings.LLM_FALLBACK_ENABLED and settings.QWEN_BASE_URL:
        try:
            secondary = QwenProvider()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to init QwenProvider (fallback disabled): %s", exc)

    if primary and secondary:
        logger.info(
            "LLM provider: Gemini (primary) + Qwen (fallback, circuit=%ds)",
            settings.LLM_CIRCUIT_BREAK_SECONDS,
        )
        return FallbackLLMProvider(primary=primary, secondary=secondary)
    if primary:
        return primary
    if secondary:
        logger.info("LLM provider: Qwen only (no Gemini key)")
        return secondary

    # Default scripted response — keeps offline dev usable.
    return FakeLLMProvider(
        script=[
            AIResponse(
                text="<thought>offline-mode: no LLM configured</thought>"
                     "<answer>当前为离线模式，未配置 GEMINI_API_KEY。</answer>"
            )
        ]
    )
