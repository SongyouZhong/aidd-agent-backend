"""LangGraph ReAct agent (design doc §7.4).

Minimal Phase-4 graph:

    START → agent_node ──tool_calls?──> tool_node ─┐
                │ no                                │
                ▼                                   │
               END  ◀─────────────────────────────  ┘

``agent_node`` renders the System Prompt fresh each turn (so dynamic
slots — time, active tools, session memory — are always current),
appends the assistant pre-fill, and asks the LLM provider for a
response. If the response contains tool calls, the graph routes to
``tool_node``; otherwise the graph terminates and the answer is
post-processed for citations.

This Phase-4 version intentionally focuses on the prompt + grounding
loop. Subagents and Auto-Compaction land in Phase 5.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.citations import Citation, inject_citations
from app.agent.context_manager import (
    CompactTrackingState,
    apply_compaction,
    maybe_compact,
)
from app.agent.llm_provider import AIResponse, FakeLLMProvider
from app.agent.prompt_renderer import render_system_prompt
from app.agent.prompts.target_discovery import INTENT_ROUTER_PROMPT
from app.core.config import settings
from app.tools import default_registry, tool_search


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    hot_loaded: set[str]
    session_memory: str
    citations: list[Citation]
    final_text: str
    # Phase 5 additions
    extra_tools: dict[str, Any]      # name -> StructuredTool (e.g. deep_research_agent)
    compact_tracking: CompactTrackingState
    model: str
    session_id: str | None
    target_query: str | None  # set by intent_router when target discovery is triggered


def _strip_system(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if not isinstance(m, SystemMessage)]


def _execute_tool_call(
    name: str,
    args: dict[str, Any],
    extra_tools: dict[str, Any] | None = None,
) -> Any:
    """Look up + invoke a registered tool by name.

    Returns either a string (sync tool) or an awaitable (async tool).
    The caller is responsible for awaiting if needed.
    """
    if extra_tools and name in extra_tools:
        tool = extra_tools[name]
        if getattr(tool, "coroutine", None) is not None:
            return tool.ainvoke(args)
        return tool.invoke(args)
    if name == "tool_search":
        return tool_search.invoke(args)
    impl = default_registry.get(name)
    if impl is None:
        return f"[error] tool '{name}' is not loaded"
    # Async tools (decorated with @tool on an async def) must use ainvoke.
    if getattr(impl, "coroutine", None) is not None:
        return impl.ainvoke(args)
    return impl.invoke(args)



def build_agent(provider: Any):
    """Compile a fresh graph bound to ``provider``."""

    async def agent_node(state: AgentState) -> dict[str, Any]:
        # --- Phase 5: Auto-Compaction guard (design doc §9) ---------
        tracking = state.get("compact_tracking") or CompactTrackingState()
        model = state.get("model") or settings.GEMINI_MODEL

        async def _summarizer(msgs: list[BaseMessage]) -> str:
            resp = await provider.generate(messages=msgs, tools=None)
            return resp.text

        compact_result = await maybe_compact(
            state["messages"],
            model=model,
            tracking=tracking,
            summarizer=_summarizer,
            session_memory=state.get("session_memory"),
        )
        compacted_messages = state["messages"]
        if compact_result is not None:
            compacted_messages = apply_compaction(
                list(state["messages"]), compact_result
            )

        hot = state.get("hot_loaded") or set()
        active_tools = default_registry.bind_active(hot_loaded=hot)
        extra_tools: dict[str, Any] = state.get("extra_tools") or {}
        active_names = (
            ["tool_search"]
            + [t.name for t in active_tools]
            + list(extra_tools.keys())
        )

        system = SystemMessage(
            content=render_system_prompt(
                active_tools=active_names,
                hot_loaded=hot,
                session_memory=state.get("session_memory"),
            )
        )
        history = _strip_system(compacted_messages)

        result: AIResponse = await provider.generate(
            messages=[system, *history],
            tools=active_tools + list(extra_tools.values()),
        )

        ai_msg = AIMessage(
            content=result.text,
            tool_calls=[
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in result.tool_calls
            ],
        )
        ai_msg.additional_kwargs["grounding_metadata"] = result.grounding_metadata

        out: dict[str, Any] = {"messages": [ai_msg], "compact_tracking": tracking}
        if compact_result is not None:
            # Use RemoveMessage to drop the entire prior history under
            # the ``add_messages`` reducer, then append summary + reply.
            removals: list[BaseMessage] = []
            kept_ids = {id(m) for m in compact_result.messages_to_keep}
            for m in state["messages"]:
                if id(m) in kept_ids:
                    continue
                msg_id = getattr(m, "id", None)
                if msg_id:
                    removals.append(RemoveMessage(id=msg_id))
            out["messages"] = [
                *removals,
                *compact_result.summary_messages,
                ai_msg,
            ]
        return out

    async def tool_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        new_msgs: list[BaseMessage] = []
        new_hot = set(state.get("hot_loaded") or set())
        extra_tools = state.get("extra_tools") or {}

        for tc in getattr(last, "tool_calls", []) or []:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            result = _execute_tool_call(name, args, extra_tools=extra_tools)
            if hasattr(result, "__await__"):
                content = await result
            else:
                content = result
            new_msgs.append(
                ToolMessage(content=str(content), name=name, tool_call_id=tc["id"])
            )

            # Auto-mount tools surfaced by tool_search (design §7.3.1).
            if name == "tool_search":
                try:
                    import json

                    payload = json.loads(content)
                    for m in payload.get("matches", []):
                        if m.get("name"):
                            new_hot.add(m["name"])
                except Exception:
                    pass

        return {"messages": new_msgs, "hot_loaded": new_hot}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tool"
        return "finalize"

    async def finalize_node(state: AgentState) -> dict[str, Any]:
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai is None:
            return {"final_text": "", "citations": []}
        gm = last_ai.additional_kwargs.get("grounding_metadata")
        text, citations = inject_citations(last_ai.content, gm)
        return {"final_text": text, "citations": citations}

    # --- intent router node ----------------------------------------
    async def intent_router_node(state: AgentState) -> dict[str, Any]:
        """Classify the latest human message: target_discovery or general."""
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {"target_query": None}
        resp = await provider.generate(
            messages=[
                SystemMessage(content=INTENT_ROUTER_PROMPT),
                HumanMessage(content=str(last_human.content)),
            ],
            tools=None,
        )
        try:
            m = re.search(r"\{.*?\}", resp.text.strip(), re.DOTALL)
            if m:
                data = json.loads(m.group())
                if data.get("route") == "target_discovery" and data.get("target_query"):
                    return {"target_query": str(data["target_query"])}
        except Exception:
            pass
        return {"target_query": None}

    # --- target discovery node --------------------------------------
    async def target_discovery_node(state: AgentState) -> dict[str, Any]:
        """Run the 5-node target discovery sub-graph and return a report message."""
        from app.agent.target_discovery_graph import build_target_discovery_graph

        query = state.get("target_query") or ""
        td_graph = build_target_discovery_graph(provider)
        final_state = await td_graph.ainvoke({
            "target_query": query,
            "messages": [],
            "sub_results": {},
            "notes": [],
            "final_report": {},
        })
        report = final_state.get("final_report") or {}
        report_text = (
            f"## 靶点发现报告：{query}\n\n"
            f"```json\n{json.dumps(report, ensure_ascii=False, indent=2)}\n```"
        )
        return {"messages": [AIMessage(content=report_text)]}

    def route_intent(state: AgentState) -> str:
        return "target_discovery" if state.get("target_query") else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("target_discovery", target_discovery_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {"target_discovery": "target_discovery", "agent": "agent"},
    )
    graph.add_edge("target_discovery", "finalize")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tool": "tool", "finalize": "finalize"},
    )
    graph.add_edge("tool", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


# --- ergonomic one-shot helper ----------------------------------------

async def run_once(
    provider: Any,
    user_prompt: str,
    *,
    session_memory: str | None = None,
    extra_tools: dict[str, Any] | None = None,
    model: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run a single user turn end-to-end. Returns the final state."""
    agent = build_agent(provider)
    initial: AgentState = {
        "messages": [HumanMessage(content=user_prompt)],
        "hot_loaded": set(),
        "session_memory": session_memory or "",
        "extra_tools": extra_tools or {},
        "compact_tracking": CompactTrackingState(),
        "model": model or settings.GEMINI_MODEL,
        "session_id": session_id,
    }
    return await agent.ainvoke(initial)
