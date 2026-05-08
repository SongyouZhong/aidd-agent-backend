"""Target-Discovery LangGraph sub-graph (design doc §7.4 — Phase 6).

Fixed-node pipeline:

    START
      → literature_node    (PubMed / arXiv)
      → composition_node   (UniProt / PDB / AlphaFold / InterPro)
      → function_node      (OpenTargets / Monarch / QuickGO)
      → pathway_node       (KEGG / Reactome / STRING)
      → drugs_node         (ChEMBL target-act / PubChem / GtoPdb / peptides)
      → synthesize_node    (no tools — assembles TargetReport)
      → END

Each node runs a bounded ReAct loop (max ``MAX_NODE_STEPS`` LLM turns)
binding only its allowed tool subset. On any error / timeout the node
stores a ``notes`` entry and the pipeline still proceeds — the design
goal is "always produce a partial report" (plan §Further Considerations).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import pathlib
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.llm_provider import AIResponse
from app.agent.prompts.target_discovery import (
    COMPOSITION_NODE_PROMPT,
    DRUGS_NODE_PROMPT,
    FUNCTION_NODE_PROMPT,
    LITERATURE_NODE_PROMPT,
    PATHWAY_NODE_PROMPT,
    SYNTHESIZE_PROMPT,
)
from app.tools import default_registry

logger = logging.getLogger(__name__)


MAX_NODE_STEPS = 10
NODE_TIMEOUT_SECONDS = 120.0

LOGS_DIR = pathlib.Path("logs/target_discovery")


# Tool subsets per node — names must match registered tool names.
LITERATURE_TOOLS = ["query_pubmed", "query_arxiv"]
COMPOSITION_TOOLS = [
    "query_uniprot",
    "query_pdb",
    "query_pdb_identifiers",
    "query_alphafold",
    "query_interpro",
]
FUNCTION_TOOLS = ["query_opentarget", "query_monarch", "query_quickgo"]
PATHWAY_TOOLS = ["query_kegg", "query_reactome", "query_stringdb", "query_graph_schema", "query_wikipathways_graph"]
DRUGS_TOOLS = [
    "query_chembl_target_activities",
    "query_pubchem",
    "query_gtopdb",
    "query_chembl_peptides",
]


class TargetDiscoveryState(TypedDict, total=False):
    target_query: str
    messages: Annotated[list[BaseMessage], add_messages]
    sub_results: dict[str, Any]
    notes: list[str]
    final_report: dict[str, Any]


# --- helpers ----------------------------------------------------------


def _resolve_tools(names: list[str]) -> list[Any]:
    out = []
    for n in names:
        impl = default_registry.get(n)
        if impl is not None:
            out.append(impl)
    return out


def _render(template: str, **kwargs: Any) -> str:
    text = template
    for k, v in kwargs.items():
        text = text.replace("{{ " + k + " }}", str(v))
    return text


async def _invoke_tool(name: str, args: dict[str, Any]) -> str:
    impl = default_registry.get(name)
    if impl is None:
        return f"[error] tool '{name}' not loaded"
    if getattr(impl, "coroutine", None) is not None:
        result = await impl.ainvoke(args)
    else:
        result = await asyncio.to_thread(impl.invoke, args)
    return str(result)


def _extract_answer_json(text: str) -> dict[str, Any] | None:
    """Pull the first ``{...}`` JSON object out of an <answer> block or raw text.

    Handles:
    - ``<answer>...</answer>`` tags
    - Markdown ````json ... ``` `` code fences
    - Bare ``{...}`` in the text
    """
    if not text:
        return None
    blob = text
    if "<answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>", start)
        blob = text[start:end] if end > start else text[start:]
    # Try markdown code fence first (```json ... ``` or ``` ... ```)
    code_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.DOTALL)
    if code_fence:
        try:
            return json.loads(code_fence.group(1))
        except Exception:
            pass
    # Greedy match the outermost {...}
    first = blob.find("{")
    last = blob.rfind("}")
    if first == -1 or last <= first:
        return None
    try:
        return json.loads(blob[first : last + 1])
    except Exception:
        return None


async def _run_node_loop(
    *,
    provider: Any,
    messages: list[BaseMessage],
    tool_names: list[str],
    max_steps: int = MAX_NODE_STEPS,
) -> str:
    """Bounded ReAct loop scoped to ``tool_names``. Mutates `messages` in place. Returns last_text."""
    tools = _resolve_tools(tool_names)
    last_text = ""
    for _ in range(max_steps):
        resp: AIResponse = await provider.generate(messages=messages, tools=tools)
        last_text = resp.text or ""
        ai_msg = AIMessage(
            content=last_text,
            tool_calls=[
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in resp.tool_calls
            ],
        )
        messages.append(ai_msg)
        if not resp.tool_calls:
            break
        for tc in resp.tool_calls:
            try:
                tool_out = await _invoke_tool(tc.name, tc.args)
            except Exception as exc:
                tool_out = f"[tool error] {exc}"
            messages.append(
                ToolMessage(content=tool_out, name=tc.name, tool_call_id=tc.id)
            )
    # Safety net: if loop exhausted or last text has no parseable JSON,
    # make one final call without tools to force structured output.
    if _extract_answer_json(last_text) is None:
        messages.append(
            HumanMessage(
                content=(
                    "Based on all the tool query results above, please directly output the final JSON summary in the "
                    "<answer>...</answer> format, and do not call any more tools."
                )
            )
        )
        try:
            final_resp: AIResponse = await provider.generate(messages=messages, tools=None)
            if final_resp.text:
                last_text = final_resp.text
        except Exception as exc:
            logger.warning("Forced final summary call failed: %s", exc)
    return last_text


async def _safe_node(
    *,
    name: str,
    provider: Any,
    target_query: str,
    template: str,
    tool_names: list[str],
    prior_context: str | None = None,
    log_dir: pathlib.Path | None = None,
    timeout_seconds: float = NODE_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], list[str]]:
    """Run one node, catch exceptions and timeouts, return (result, notes)."""
    sys_prompt = _render(template, target_query=target_query)
    user_prompt = f"Start executing node [{name}], target: {target_query}."
    if prior_context:
        user_prompt += (
            f"\n\n❗ MANDATORY CONSTRAINT: The following are verified UniProt accessions. "
            f"You MUST strictly use these values when calling any tool's uniprot_id parameter. "
            f"The use of any other accession (such as Q13168, etc., which are incorrect) is prohibited:\n{prior_context}"
        )
    messages: list[BaseMessage] = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        last_text = await asyncio.wait_for(
            _run_node_loop(
                provider=provider,
                messages=messages,
                tool_names=tool_names,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Node [%s] timed out (>%.0fs). Attempting fallback summarization.", name, timeout_seconds)
        notes_out = [f"Node [{name}] timed out (>{timeout_seconds:.0f}s), attempted partial summarization."]
        messages.append(
            HumanMessage(
                content=(
                    "Execution time limit reached. Based on the tool query results gathered so far, "
                    "please directly output the final JSON summary in the <answer>...</answer> format. "
                    "Do not call any more tools."
                )
            )
        )
        try:
            final_resp: AIResponse = await asyncio.wait_for(
                provider.generate(messages=messages, tools=None),
                timeout=60.0,
            )
            last_text = final_resp.text or ""
            parsed = _extract_answer_json(last_text)
            if parsed is None:
                notes_out.append(f"Node [{name}] partial summarization did not output parseable JSON.")
                parsed = {}
        except Exception as exc:
            logger.warning("Node [%s] fallback summarization failed: %s", name, exc)
            notes_out.append(f"Node [{name}] fallback summarization failed: {exc!r}")
            last_text = ""
            parsed = {}
            
        await _write_node_log(log_dir, name, target_query, messages, last_text, parsed, notes_out)
        return parsed, notes_out
    except Exception as exc:
        logger.exception("Node %s failed", name)
        notes_out = [f"Node [{name}] error: {exc!r}"]
        await _write_node_log(log_dir, name, target_query, messages, "", {}, notes_out)
        return {}, notes_out
    parsed = _extract_answer_json(last_text)
    if parsed is None:
        notes_out = [f"Node [{name}] did not output parseable JSON."]
        await _write_node_log(log_dir, name, target_query, messages, last_text, {}, notes_out)
        return {}, notes_out
    await _write_node_log(log_dir, name, target_query, messages, last_text, parsed, [])
    return parsed, []


async def _write_node_log(
    log_dir: pathlib.Path | None,
    name: str,
    target_query: str,
    messages: list[BaseMessage],
    raw_output: str,
    parsed: dict[str, Any],
    notes: list[str],
) -> None:
    """Serialize a node's full conversation + result to a JSON file."""
    if log_dir is None:
        return
    try:
        await asyncio.to_thread(log_dir.mkdir, parents=True, exist_ok=True)
        payload = {
            "node": name,
            "target": target_query,
            "messages": _serialize_messages(messages),
            "raw_output": raw_output,
            "parsed": parsed,
            "notes": notes,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        await asyncio.to_thread((log_dir / f"{name}.json").write_text, text)
    except Exception as exc:
        logger.warning("Failed to write node log for %s: %s", name, exc)


# --- logging helpers --------------------------------------------------


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to plain dicts for JSON serialization."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            d: dict[str, Any] = {"role": "assistant", "content": m.content}
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                d["tool_calls"] = [
                    {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
                    for tc in tcs
                ]
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "name": m.name,
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
        else:
            out.append({"role": type(m).__name__, "content": str(m.content)})
    return out


# --- node definitions -------------------------------------------------


def _resolved_accession_context(sub_results: dict[str, Any]) -> str | None:
    """Extract a short context string from the composition node's output.

    Returns e.g. "UniProt accession: Q13148 (gene: TARDBP)" so that
    downstream nodes can call tools with the correct IDs rather than
    relying on the LLM to guess them.
    """
    comp = sub_results.get("composition") or {}
    proteins = comp.get("proteins") or []
    if not proteins:
        return None
    seen: set[str] = set()
    parts = []
    for p in proteins:
        acc = p.get("accession")
        gene = p.get("gene")
        if acc and acc not in seen:
            seen.add(acc)
            parts.append(f"UniProt accession: {acc}" + (f" (gene: {gene})" if gene else ""))
    return "\n".join(parts) if parts else None


def build_target_discovery_graph(provider: Any):
    """Compile the target-discovery sub-graph bound to ``provider``."""

    async def literature_node(state: TargetDiscoveryState) -> dict[str, Any]:
        # Create a per-run log directory shared by all nodes in this run.
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        slug = re.sub(r"[^\w\-]", "_", state["target_query"])[:40].strip("_")
        run_log_dir = LOGS_DIR / f"{ts}_{slug}"
        await asyncio.to_thread(run_log_dir.mkdir, parents=True, exist_ok=True)
        result, notes = await _safe_node(
            name="literature",
            provider=provider,
            target_query=state["target_query"],
            template=LITERATURE_NODE_PROMPT,
            tool_names=LITERATURE_TOOLS,
            log_dir=run_log_dir,
        )
        sub = dict(state.get("sub_results") or {})
        sub["literature"] = result
        sub["_run_log_dir"] = str(run_log_dir)
        return {"sub_results": sub, "notes": (state.get("notes") or []) + notes}

    async def composition_node(state: TargetDiscoveryState) -> dict[str, Any]:
        run_log_dir = pathlib.Path(state.get("sub_results", {}).get("_run_log_dir") or str(LOGS_DIR / "unknown"))
        result, notes = await _safe_node(
            name="composition",
            provider=provider,
            target_query=state["target_query"],
            template=COMPOSITION_NODE_PROMPT,
            tool_names=COMPOSITION_TOOLS,
            log_dir=run_log_dir,
        )
        sub = dict(state.get("sub_results") or {})
        sub["composition"] = result
        return {"sub_results": sub, "notes": (state.get("notes") or []) + notes}

    async def function_node(state: TargetDiscoveryState) -> dict[str, Any]:
        run_log_dir = pathlib.Path(state.get("sub_results", {}).get("_run_log_dir") or str(LOGS_DIR / "unknown"))
        result, notes = await _safe_node(
            name="function",
            provider=provider,
            target_query=state["target_query"],
            template=FUNCTION_NODE_PROMPT,
            tool_names=FUNCTION_TOOLS,
            prior_context=_resolved_accession_context(state.get("sub_results") or {}),
            log_dir=run_log_dir,
        )
        sub = dict(state.get("sub_results") or {})
        sub["function"] = result
        return {"sub_results": sub, "notes": (state.get("notes") or []) + notes}

    async def pathway_node(state: TargetDiscoveryState) -> dict[str, Any]:
        run_log_dir = pathlib.Path(state.get("sub_results", {}).get("_run_log_dir") or str(LOGS_DIR / "unknown"))
        result, notes = await _safe_node(
            name="pathway",
            provider=provider,
            target_query=state["target_query"],
            template=PATHWAY_NODE_PROMPT,
            tool_names=PATHWAY_TOOLS,
            prior_context=_resolved_accession_context(state.get("sub_results") or {}),
            log_dir=run_log_dir,
            timeout_seconds=300.0,
        )
        sub = dict(state.get("sub_results") or {})
        sub["pathway"] = result
        return {"sub_results": sub, "notes": (state.get("notes") or []) + notes}

    async def drugs_node(state: TargetDiscoveryState) -> dict[str, Any]:
        run_log_dir = pathlib.Path(state.get("sub_results", {}).get("_run_log_dir") or str(LOGS_DIR / "unknown"))
        result, notes = await _safe_node(
            name="drugs",
            provider=provider,
            target_query=state["target_query"],
            template=DRUGS_NODE_PROMPT,
            tool_names=DRUGS_TOOLS,
            prior_context=_resolved_accession_context(state.get("sub_results") or {}),
            log_dir=run_log_dir,
            timeout_seconds=300.0,
        )
        sub = dict(state.get("sub_results") or {})
        sub["drugs"] = result
        return {"sub_results": sub, "notes": (state.get("notes") or []) + notes}

    async def synthesize_node(state: TargetDiscoveryState) -> dict[str, Any]:
        sub_results = state.get("sub_results") or {}
        sub_json = json.dumps(sub_results, ensure_ascii=False, indent=2)
        sys_prompt = _render(SYNTHESIZE_PROMPT, sub_results_json=sub_json)
        user_prompt = (
            f"Target: {state['target_query']}. Please integrate the sub-node results above and output the TargetReport."
        )
        try:
            resp = await asyncio.wait_for(
                provider.generate(
                    messages=[
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=user_prompt),
                    ],
                    tools=None,
                ),
                timeout=300.0,
            )
            text = resp.text
        except asyncio.TimeoutError:
            text = ""

        report = _extract_answer_json(text) or {}
        # Always carry forward node-level notes.
        existing_notes = list(report.get("notes") or [])
        existing_notes.extend(state.get("notes") or [])
        report["notes"] = existing_notes
        # Ensure target field is at minimum set.
        report.setdefault(
            "target",
            {
                "name": state["target_query"],
                "gene_symbol": None,
                "uniprot_ids": [],
                "organism": "Homo sapiens",
            },
        )
        # Write final consolidated report to the run log directory.
        run_log_dir_str = (state.get("sub_results") or {}).get("_run_log_dir")
        if run_log_dir_str:
            try:
                run_log_dir = pathlib.Path(run_log_dir_str)
                await asyncio.to_thread(run_log_dir.mkdir, parents=True, exist_ok=True)
                text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
                await asyncio.to_thread(
                    (run_log_dir / "final_report.json").write_text, text
                )
                logger.info("Run log saved to %s", run_log_dir)
            except Exception as exc:
                logger.warning("Failed to write final report log: %s", exc)
        return {"final_report": report}

    graph = StateGraph(TargetDiscoveryState)
    graph.add_node("literature", literature_node)
    graph.add_node("composition", composition_node)
    graph.add_node("function", function_node)
    graph.add_node("pathway", pathway_node)
    graph.add_node("drugs", drugs_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "literature")
    graph.add_edge("literature", "composition")
    graph.add_edge("composition", "function")
    graph.add_edge("function", "pathway")
    graph.add_edge("pathway", "drugs")
    graph.add_edge("drugs", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


async def run_target_discovery(
    provider: Any, target_query: str
) -> dict[str, Any]:
    """One-shot helper. Returns the final ``TargetReport`` dict."""
    graph = build_target_discovery_graph(provider)
    initial: TargetDiscoveryState = {
        "target_query": target_query,
        "messages": [],
        "sub_results": {},
        "notes": [],
        "final_report": {},
    }
    final_state = await graph.ainvoke(initial)
    return final_state.get("final_report") or {}
