"""LLM provider — wraps Gemini (google-genai) + offline fake for tests.

Public contract:
    ``await provider.generate(messages, tools=...) -> AIResponse``

``AIResponse`` carries the answer text, optional grounding metadata, and
the list of tool calls the model wants to perform. Both real and fake
providers obey the same contract so the LangGraph node is identical.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
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


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""
    type: str  # "text" | "tool_call" | "thinking"
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


# --- Round-scoped Failure Tracking -------------------------------------

_failed_models_in_round: contextvars.ContextVar[set[str]] = contextvars.ContextVar("failed_models_in_round")

def get_failed_models() -> set[str]:
    """Get the set of failed models for the current dialogue round."""
    try:
        return _failed_models_in_round.get()
    except LookupError:
        s: set[str] = set()
        _failed_models_in_round.set(s)
        return s

def reset_failed_models() -> None:
    """Reset the failed models tracking for a new dialogue round."""
    _failed_models_in_round.set(set())


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
        max_tokens: int | None = None,
    ) -> AIResponse:
        if not self._script:
            return AIResponse(text="<answer>(no scripted response)</answer>")
        return self._script.pop(0)

    async def stream(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        resp = await self.generate(messages, tools)
        # Fake streaming: yield entire text as one chunk.
        if resp.text:
            yield StreamChunk(type="text", content=resp.text)
        for tc in resp.tool_calls:
            yield StreamChunk(
                type="tool_call", tool_name=tc.name,
                tool_args=tc.args, tool_call_id=tc.id,
            )


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
        self.model = model or settings.GEMINI_MODELS.split(",")[0].strip()
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
        max_tokens: int | None = None,
    ) -> AIResponse:
        system_text, contents = self._to_gemini_contents(messages)
        from google.genai import types  # type: ignore

        config_kwargs: dict[str, Any] = {}
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

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

    async def stream(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming variant: yields chunks as the model generates."""
        system_text, contents = self._to_gemini_contents(messages)
        from google.genai import types  # type: ignore

        config_kwargs: dict[str, Any] = {}
        if system_text:
            config_kwargs["system_instruction"] = system_text

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
            gemini_tools.append(types.Tool(google_search=types.GoogleSearch()))
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response_stream = await asyncio.to_thread(
            lambda: list(self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ))
        )

        tc_counter = 0
        for chunk in response_stream:
            candidates = getattr(chunk, "candidates", None) or []
            if not candidates:
                continue
            content = getattr(candidates[0], "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                if hasattr(part, "text") and part.text:
                    yield StreamChunk(type="text", content=part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tc_counter += 1
                    yield StreamChunk(
                        type="tool_call",
                        tool_name=fc.name,
                        tool_args=dict(fc.args) if fc.args else {},
                        tool_call_id=f"call-{fc.name}-{tc_counter}",
                    )


# --- Qwen provider (local vLLM, OpenAI-compatible) -------------------

class OpenAICompatibleProvider:
    """Base for OpenAI-compatible providers (vLLM, DeepSeek, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        from openai import AsyncOpenAI  # type: ignore

        self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def _get_extra_body(self) -> dict[str, Any] | None:
        """Return provider-specific extensions (e.g. thinking mode)."""
        return None

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
        max_tokens: int | None = None,
    ) -> AIResponse:
        oa_messages = self._to_openai_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oa_messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        extra = self._get_extra_body()
        if extra:
            kwargs["extra_body"] = extra

        if tools:
            oa_tools = self._langchain_tools_to_openai(tools)
            if oa_tools:
                kwargs["tools"] = oa_tools

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "")
        if reasoning:
            text = f"<thought>\n{reasoning}\n</thought>\n\n{text}"

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

    async def stream(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        oa_messages = self._to_openai_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oa_messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 4096,
            "stream": True,
        }
        extra = self._get_extra_body()
        if extra:
            kwargs["extra_body"] = extra

        if tools:
            oa_tools = self._langchain_tools_to_openai(tools)
            if oa_tools:
                kwargs["tools"] = oa_tools

        stream_resp = await self._client.chat.completions.create(**kwargs)

        tc_buffer: dict[int, dict[str, Any]] = {}
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", "")
            if reasoning:
                yield StreamChunk(type="thinking", content=reasoning)
            if delta.content:
                yield StreamChunk(type="text", content=delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_buffer:
                        tc_buffer[idx] = {
                            "id": tc.id or f"call-{uuid.uuid4().hex[:8]}",
                            "name": tc.function.name or "",
                            "arguments": "",
                        }
                    if tc.function.name:
                        tc_buffer[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tc_buffer[idx]["arguments"] += tc.function.arguments

        for buf in tc_buffer.values():
            try:
                args = json.loads(buf["arguments"] or "{}")
            except (TypeError, ValueError):
                args = {}
            yield StreamChunk(
                type="tool_call",
                tool_name=buf["name"],
                tool_args=args,
                tool_call_id=buf["id"],
            )


class QwenProvider(OpenAICompatibleProvider):
    """Local Qwen3 served by vLLM via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url or settings.QWEN_BASE_URL,
            model=model or settings.QWEN_MODEL,
            api_key=api_key or settings.QWEN_API_KEY,
        )
        self.enable_thinking = enable_thinking

    def _get_extra_body(self) -> dict[str, Any] | None:
        if self.enable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": True}}
        return None


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider (V4 compatible)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool = True,  # V4 defaults to thinking enabled
    ) -> None:
        super().__init__(
            base_url=base_url or settings.DEEPSEEK_BASE_URL,
            model=model or settings.DEEPSEEK_MODELS.split(",")[0],
            api_key=api_key or settings.DEEPSEEK_API_KEY,
        )
        self.enable_thinking = enable_thinking

    def _get_extra_body(self) -> dict[str, Any] | None:
        if self.enable_thinking:
            # DeepSeek V4 Thinking Mode: https://api-docs.deepseek.com/guides/thinking_mode
            return {"thinking": {"type": "enabled"}}
        return None

    @staticmethod
    def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Same as base, but for assistant turns split a leading
        ``<thought>...</thought>`` block back into ``reasoning_content``.

        DeepSeek's thinking mode requires the ``reasoning_content`` of any
        prior assistant turn to be echoed back; otherwise the API returns
        ``400 invalid_request_error``.
        """
        out = OpenAICompatibleProvider._to_openai_messages(messages)
        for msg_dict in out:
            if msg_dict.get("role") != "assistant":
                continue
            content = msg_dict.get("content")
            if not isinstance(content, str):
                continue
            stripped = content.lstrip()
            if not stripped.startswith("<thought>"):
                continue
            end = stripped.find("</thought>")
            if end == -1:
                continue
            reasoning = stripped[len("<thought>"):end].strip()
            rest = stripped[end + len("</thought>"):].lstrip("\n").lstrip()
            msg_dict["reasoning_content"] = reasoning
            msg_dict["content"] = rest or None
        return out


# --- Fallback wrapper (round-scoped circuit breaker) -------------------

class FallbackLLMProvider:
    """Primary providers with automatic fallback to a secondary.

    A *round-level* circuit breaker is applied when a primary provider returns
    a retryable infrastructure error (UNAVAILABLE / HTTP 503 / 429). The failed
    model is marked as unavailable and will be skipped for the remainder of
    the current dialogue round. If all primary providers fail, it falls back
    to the secondary provider.
    """

    def __init__(
        self,
        primaries: list[Any],
        secondary: Any | None = None,
    ) -> None:
        self.primaries = primaries
        self.secondary = secondary

    @staticmethod
    def _is_retryable_failure(exc: BaseException) -> bool:
        """Return True for infrastructure failures worth failing over on."""
        # OpenAI SDK connection errors (covers all APIConnectionError variants)
        try:
            from openai import APIConnectionError as _OAIConnErr
            if isinstance(exc, _OAIConnErr):
                return True
        except ImportError:
            pass
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
        for marker in ("DEADLINE_EXCEEDED", "TIMEOUT", "CONNECTION RESET", "CONNECTION ERROR"):
            if marker in msg:
                return True
        return False

    async def generate(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
        max_tokens: int | None = None,
    ) -> AIResponse:
        failed_models = get_failed_models()
        last_exc = None

        for provider in self.primaries:
            if getattr(provider, "model", None) in failed_models:
                continue

            try:
                return await provider.generate(messages, tools=tools, max_tokens=max_tokens)
            except Exception as exc:
                if not self._is_retryable_failure(exc):
                    raise
                model_name = getattr(provider, "model", "unknown")
                logger.warning(
                    "Primary LLM %s unavailable (%s: %s); failing over to next model",
                    model_name,
                    type(exc).__name__,
                    exc,
                )
                failed_models.add(model_name)
                last_exc = exc

        # All primaries failed or none were available
        if self.secondary:
            return await self.secondary.generate(messages, tools=tools, max_tokens=max_tokens)

        if last_exc:
            raise last_exc
        raise RuntimeError("No available LLM providers to handle the request.")

    async def stream(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming variant with fallback."""
        failed_models = get_failed_models()
        last_exc = None

        for provider in self.primaries:
            if getattr(provider, "model", None) in failed_models:
                continue

            try:
                # Probe primary with stream; if it works, stream from it.
                async for chunk in provider.stream(messages, tools=tools):
                    yield chunk
                return
            except Exception as exc:
                if not self._is_retryable_failure(exc):
                    raise
                model_name = getattr(provider, "model", "unknown")
                logger.warning(
                    "Primary LLM stream %s failed (%s: %s); falling over to next model",
                    model_name,
                    type(exc).__name__,
                    exc,
                )
                failed_models.add(model_name)
                last_exc = exc

        # All primaries failed
        if self.secondary:
            async for chunk in self.secondary.stream(messages, tools=tools):
                yield chunk
            return

        if last_exc:
            raise last_exc
        raise RuntimeError("No available LLM providers to handle the stream request.")


# --- Factory -----------------------------------------------------------

def get_default_provider() -> Any:
    """Return the configured provider with dynamic priority and optional fallback.

    Resolution order is determined by `settings.LLM_PRIORITY` (e.g., "gemini,deepseek").
    """
    if os.environ.get("AIDD_FORCE_FAKE_LLM"):
        return FakeLLMProvider(
            script=[AIResponse(text="<answer>(forced offline mode)</answer>")]
        )

    primaries = []
    priority_list = [p.strip().lower() for p in settings.LLM_PRIORITY.split(",") if p.strip()]

    for p_type in priority_list:
        if p_type == "gemini" and settings.GEMINI_API_KEY:
            model_names = [m.strip() for m in settings.GEMINI_MODELS.split(",") if m.strip()]
            for m in model_names:
                try:
                    primaries.append(GeminiProvider(model=m))
                except Exception as exc:
                    logger.warning("Failed to init GeminiProvider for %s: %s", m, exc)

        elif p_type == "deepseek" and settings.DEEPSEEK_API_KEY:
            model_names = [m.strip() for m in settings.DEEPSEEK_MODELS.split(",") if m.strip()]
            for m in model_names:
                try:
                    primaries.append(DeepSeekProvider(model=m))
                except Exception as exc:
                    logger.warning("Failed to init DeepSeekProvider for %s: %s", m, exc)

    secondary = None
    if settings.LLM_FALLBACK_ENABLED and settings.QWEN_BASE_URL:
        try:
            secondary = QwenProvider()
        except Exception as exc:
            logger.warning("Failed to init QwenProvider (fallback disabled): %s", exc)

    if primaries:
        logger.info(
            "LLM providers: %s%s",
            ", ".join(getattr(p, "model", "unknown") for p in primaries),
            " + Qwen (fallback)" if secondary else "",
        )
        return FallbackLLMProvider(primaries=primaries, secondary=secondary)

    if secondary:
        logger.info("LLM provider: Qwen only")
        return secondary

    return FakeLLMProvider(
        script=[
            AIResponse(
                text="<thought>offline-mode: no LLM configured</thought>"
                     "<answer>当前为离线模式，未配置有效 LLM。</answer>"
            )
        ]
    )


def get_graph_rag_llm() -> Any:
    """Return a *synchronous* LangChain ChatOpenAI for use with GraphCypherQAChain.

    Follows the same ``LLM_PRIORITY`` resolution as ``get_default_provider()``,
    but only considers OpenAI-compatible endpoints (DeepSeek / Qwen).
    Gemini is skipped because GraphCypherQAChain requires an OpenAI-compatible
    interface. Thinking mode is disabled — Cypher generation needs deterministic
    structured output, not reasoning traces.
    """
    from langchain_openai import ChatOpenAI  # type: ignore

    priority_list = [p.strip().lower() for p in settings.LLM_PRIORITY.split(",") if p.strip()]

    for p_type in priority_list:
        if p_type == "deepseek" and settings.DEEPSEEK_BASE_URL and settings.DEEPSEEK_API_KEY:
            model = settings.DEEPSEEK_MODELS.split(",")[0].strip()
            logger.info("GraphRAG LLM: DeepSeek %s at %s", model, settings.DEEPSEEK_BASE_URL)
            return ChatOpenAI(
                model=model,
                base_url=settings.DEEPSEEK_BASE_URL,
                api_key=settings.DEEPSEEK_API_KEY,
                temperature=0.0,
            )
        if p_type == "qwen" and settings.QWEN_BASE_URL:
            logger.info("GraphRAG LLM: Qwen %s at %s", settings.QWEN_MODEL, settings.QWEN_BASE_URL)
            return ChatOpenAI(
                model=settings.QWEN_MODEL,
                base_url=settings.QWEN_BASE_URL,
                api_key=settings.QWEN_API_KEY or "empty",
                temperature=0.0,
            )

    # Fallback: Qwen if not already tried via priority
    if settings.QWEN_BASE_URL:
        logger.info("GraphRAG LLM: Qwen %s at %s (fallback)", settings.QWEN_MODEL, settings.QWEN_BASE_URL)
        return ChatOpenAI(
            model=settings.QWEN_MODEL,
            base_url=settings.QWEN_BASE_URL,
            api_key=settings.QWEN_API_KEY or "empty",
            temperature=0.0,
        )

    raise RuntimeError(
        "No OpenAI-compatible LLM configured for GraphRAG. "
        "Set DEEPSEEK_BASE_URL+DEEPSEEK_API_KEY or QWEN_BASE_URL in .env, "
        "and ensure the corresponding entry appears in LLM_PRIORITY."
    )
