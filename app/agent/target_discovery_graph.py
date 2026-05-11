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


MAX_NODE_STEPS = 6
# Wall-clock budget per node (serial: composition; parallel: others).
NODE_TIMEOUT_SECONDS = 200.0
# Composition is the serial bottleneck (4-7 sequential tool calls). Give it
# extra headroom so a slow UniProt/PDB round doesn't cascade-fail downstream.
COMPOSITION_TIMEOUT_SECONDS = 240.0
# Cap per upstream tool call (network I/O).
TOOL_TIMEOUT_SECONDS = 30.0

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
    sub_results: Annotated[dict[str, Any], lambda a, b: {**(a or {}), **(b or {})}]
    notes: Annotated[list[str], lambda a, b: list(a or []) + list(b or [])]
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
    # Per-tool timeout: a single slow upstream API (e.g. GtoPdb 503 retries)
    # must not be allowed to consume the entire node wall-clock budget.
    try:
        if getattr(impl, "coroutine", None) is not None:
            result = await asyncio.wait_for(impl.ainvoke(args), timeout=TOOL_TIMEOUT_SECONDS)
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(impl.invoke, args),
                timeout=TOOL_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        return f"[tool timeout: {name} >{TOOL_TIMEOUT_SECONDS:.0f}s]"
    return str(result)


_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)


def _balanced_json_objects(text: str) -> list[str]:
    """Return all top-level balanced ``{...}`` substrings in ``text``.

    Tracks string literals + escapes so braces inside strings don't confuse
    the matcher. Used as the last-resort scan when no <answer>/code-fence
    block is present.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


def _extract_answer_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of an LLM response.

    Strategy (in order):
    1) Strip ``<thought>...</thought>`` blocks (their stray ``{}`` corrupted
       the prior greedy fallback).
    2) Prefer the inside of an ``<answer>...</answer>`` tag if present.
    3) Try a ```json fenced``` block.
    4) ``json.loads`` the trimmed text directly.
    5) Scan for *all* top-level balanced ``{...}`` substrings and return the
       largest one that parses (handles trailing prose after the JSON).
    """
    if not text:
        return None
    cleaned = _THOUGHT_RE.sub("", text)

    if "<answer>" in cleaned:
        start = cleaned.find("<answer>") + len("<answer>")
        end = cleaned.find("</answer>", start)
        blob = cleaned[start:end] if end > start else cleaned[start:]
    else:
        blob = cleaned

    # 3) Markdown code fence
    code_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.DOTALL)
    if code_fence:
        try:
            return json.loads(code_fence.group(1))
        except Exception:
            pass

    # 4) Direct parse
    stripped = blob.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # 5) Largest balanced object that parses
    candidates = _balanced_json_objects(blob)
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _mechanical_merge(sub_results: dict[str, Any]) -> dict[str, Any]:
    """Plan-B fallback: assemble a TargetReport by directly copying fields
    from each sub-node's ``parsed`` dict to the top level.

    Used when both the primary and retry synthesize LLM calls fail to
    produce parseable JSON. Guarantees the user always sees the data the
    sub-nodes already collected, instead of an empty report.
    """
    report: dict[str, Any] = {}
    # Field → list of sub-node keys to look in (in priority order).
    field_sources = {
        "target": ["composition"],
        "proteins": ["composition"],
        "papers": ["literature"],
        "function_narrative": ["function"],
        "disease_associations": ["function"],
        "pathways": ["pathway"],
        "small_molecule_drugs": ["drugs"],
        "peptide_drugs": ["drugs"],
        "antibody_drugs": ["drugs"],
        "data_source_gaps": ["drugs", "pathway", "function", "literature", "composition"],
    }
    for field, sources in field_sources.items():
        for src in sources:
            node_out = sub_results.get(src) or {}
            if not isinstance(node_out, dict):
                continue
            if field in node_out and node_out[field] not in (None, [], {}):
                if field == "data_source_gaps" and field in report:
                    # Concatenate gaps from all nodes.
                    if isinstance(report[field], list) and isinstance(node_out[field], list):
                        report[field] = report[field] + node_out[field]
                else:
                    report[field] = node_out[field]
                    break
    return report


def _sanitize_for_summary(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return a copy of ``messages`` that is safe to send to a no-tools LLM call.

    Ensures every ``AIMessage.tool_calls`` id has a matching following
    ``ToolMessage``. For any unmatched id, inserts a placeholder
    ``ToolMessage(content='[cancelled: no tool result captured]')``.
    If the trailing message is itself an ``AIMessage`` with unfulfilled
    tool_calls, placeholders are appended at the end.

    This protects against OpenAI/DeepSeek's strict validation:
    "An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'."
    """
    out: list[BaseMessage] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        out.append(m)
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            expected_ids = [tc["id"] for tc in m.tool_calls]
            seen_ids: set[str] = set()
            j = i + 1
            while j < n and isinstance(messages[j], ToolMessage):
                tm = messages[j]
                out.append(tm)
                seen_ids.add(tm.tool_call_id)
                j += 1
            for tc in m.tool_calls:
                if tc["id"] not in seen_ids:
                    out.append(
                        ToolMessage(
                            content="[cancelled: no tool result captured]",
                            name=tc["name"],
                            tool_call_id=tc["id"],
                        )
                    )
            i = j
        else:
            i += 1
    return out


async def _run_node_loop(
    *,
    provider: Any,
    messages: list[BaseMessage],
    tool_names: list[str],
    max_steps: int = MAX_NODE_STEPS,
) -> str:
    """Bounded ReAct loop scoped to ``tool_names``. Mutates `messages` in place. Returns last_text.

    Each step uses an *atomic commit* discipline: the assistant message and
    its tool replies are buffered locally, and only appended to ``messages``
    once the entire fan-out completes. If the step is cancelled
    (``asyncio.wait_for`` timeout), the partial step is discarded so
    ``messages`` always remains a well-formed ReAct sequence.
    """
    tools = _resolve_tools(tool_names)
    last_text = ""
    # Per-loop (tool_name, normalized_args) cache: identical re-queries
    # within the same node return the cached ToolMessage content instead of
    # hitting the network again. Eliminates wasteful patterns like
    # `target="Q13148"` followed by `target="TARDBP"` (same ChEMBL hit).
    call_cache: dict[tuple[str, str], str] = {}
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
        if not resp.tool_calls:
            # No tool fan-out: safe to commit immediately.
            messages.append(ai_msg)
            break
        # Atomic commit: build the full ToolMessage list before touching `messages`.
        tool_msgs: list[ToolMessage] = []
        try:
            for tc in resp.tool_calls:
                cache_key = (tc.name, json.dumps(tc.args, sort_keys=True, default=str))
                if cache_key in call_cache:
                    tool_out = call_cache[cache_key] + "\n[note: cached duplicate call]"
                else:
                    try:
                        tool_out = await _invoke_tool(tc.name, tc.args)
                    except Exception as exc:
                        tool_out = f"[tool error] {exc}"
                    call_cache[cache_key] = tool_out
                tool_msgs.append(
                    ToolMessage(content=tool_out, name=tc.name, tool_call_id=tc.id)
                )
        except asyncio.CancelledError:
            # Step interrupted mid-flight; discard partial step entirely so
            # `messages` remains a valid (ai_msg + matching tool_msgs) sequence.
            raise
        messages.append(ai_msg)
        messages.extend(tool_msgs)
    # Safety net: if loop exhausted or last text has no parseable JSON,
    # make one final call without tools to force structured output.
    if _extract_answer_json(last_text) is None:
        safe_messages = _sanitize_for_summary(messages)
        safe_messages.append(
            HumanMessage(
                content=(
                    "Based on all the tool query results above, please directly output the final JSON summary in the "
                    "<answer>...</answer> format, and do not call any more tools."
                )
            )
        )
        try:
            final_resp: AIResponse = await provider.generate(messages=safe_messages, tools=None)
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
        # Sanitize history before the no-tools fallback call: a wait_for cancel
        # may have left an AIMessage with tool_calls that lack matching
        # ToolMessage replies, which OpenAI/DeepSeek will reject with a 400.
        safe_messages = _sanitize_for_summary(messages)
        safe_messages.append(
            HumanMessage(
                content=(
                    "Execution time limit reached. Based on the tool query results gathered so far, "
                    "please directly output the final JSON summary in the <answer>...</answer> format. "
                    "Do not call any more tools."
                )
            )
        )
        try:
            final_resp: AIResponse = await provider.generate(messages=safe_messages, tools=None)
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

    async def composition_node(state: TargetDiscoveryState) -> dict[str, Any]:
        # Composition is now the FIRST node so downstream parallel nodes can
        # consume verified UniProt + gene_symbol via prior_context.
        # It also owns creation of the per-run log directory.
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        slug = re.sub(r"[^\w\-]", "_", state["target_query"])[:40].strip("_")
        run_log_dir = LOGS_DIR / f"{ts}_{slug}"
        await asyncio.to_thread(run_log_dir.mkdir, parents=True, exist_ok=True)
        result, notes = await _safe_node(
            name="composition",
            provider=provider,
            target_query=state["target_query"],
            template=COMPOSITION_NODE_PROMPT,
            tool_names=COMPOSITION_TOOLS,
            log_dir=run_log_dir,
            timeout_seconds=COMPOSITION_TIMEOUT_SECONDS,
        )
        return {
            "sub_results": {"composition": result, "_run_log_dir": str(run_log_dir)},
            "notes": notes,
        }

    async def literature_node(state: TargetDiscoveryState) -> dict[str, Any]:
        run_log_dir = pathlib.Path(state.get("sub_results", {}).get("_run_log_dir") or str(LOGS_DIR / "unknown"))
        result, notes = await _safe_node(
            name="literature",
            provider=provider,
            target_query=state["target_query"],
            template=LITERATURE_NODE_PROMPT,
            tool_names=LITERATURE_TOOLS,
            # Pass gene_symbol so PubMed queries can use `TARDBP[gene]`
            # rather than free-text "TDP-43 highly cited review".
            prior_context=_resolved_accession_context(state.get("sub_results") or {}),
            log_dir=run_log_dir,
        )
        return {"sub_results": {"literature": result}, "notes": notes}

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
        return {"sub_results": {"function": result}, "notes": notes}

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
            timeout_seconds=180.0,
        )
        return {"sub_results": {"pathway": result}, "notes": notes}

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
            timeout_seconds=180.0,
        )
        return {"sub_results": {"drugs": result}, "notes": notes}

    async def synthesize_node(state: TargetDiscoveryState) -> dict[str, Any]:
        sub_results = state.get("sub_results") or {}
        # Drop internal-only keys (e.g. _run_log_dir) from what the LLM sees.
        sub_results_for_llm = {
            k: v for k, v in sub_results.items() if not k.startswith("_")
        }
        sub_json = json.dumps(sub_results_for_llm, ensure_ascii=False, indent=2)
        sys_prompt = _render(SYNTHESIZE_PROMPT, sub_results_json=sub_json)
        user_prompt = (
            f"Target: {state['target_query']}. Please integrate the sub-node results above and output the TargetReport."
        )
        synth_messages: list[BaseMessage] = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt),
        ]
        synth_notes: list[str] = []
        text = ""
        try:
            resp = await asyncio.wait_for(
                provider.generate(messages=synth_messages, tools=None, max_tokens=8192),
                timeout=300.0,
            )
            text = resp.text or ""
        except asyncio.TimeoutError:
            synth_notes.append("Synthesize node primary LLM call timed out (>300s).")
        except Exception as exc:
            logger.warning("Synthesize primary LLM call failed: %s", exc)
            synth_notes.append(f"Synthesize primary LLM call failed: {exc!r}")

        report = _extract_answer_json(text)

        # Plan-B fallback: when JSON parse fails, retry once with a much
        # stricter "output ONLY the JSON inside <answer>" prompt; if that
        # still fails, mechanically merge sub_results so the user always
        # sees whatever data the sub-nodes produced.
        if report is None:
            synth_notes.append(
                "Synthesize: primary output not parseable as JSON; retrying with strict prompt."
            )
            retry_user = (
                "Your previous response could not be parsed as JSON. "
                "Output ONLY the final TargetReport JSON wrapped in <answer>...</answer>. "
                "Do not include <thought>, prose, markdown fences, or any text outside the tags. "
                "The JSON must conform to the schema in the system prompt."
            )
            retry_messages: list[BaseMessage] = list(synth_messages)
            # Strip <thought>...</thought> from the previous text before
            # re-feeding: DeepSeek (and other reasoning models) treat thought
            # tags as ``reasoning_content`` and strip them from ``content``.
            # If we append a thought-only AIMessage, the API rejects the
            # request with 400 "content or tool_calls must be set".
            cleaned_prev = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
            if cleaned_prev:
                retry_messages.append(AIMessage(content=cleaned_prev))
            retry_messages.append(HumanMessage(content=retry_user))
            try:
                retry_resp = await asyncio.wait_for(
                    provider.generate(messages=retry_messages, tools=None, max_tokens=8192),
                    timeout=180.0,
                )
                retry_text = retry_resp.text or ""
                report = _extract_answer_json(retry_text)
                if report is not None:
                    text = retry_text  # log the successful retry
                    synth_notes.append("Synthesize: strict-prompt retry succeeded.")
                else:
                    synth_notes.append("Synthesize: strict-prompt retry still not parseable.")
                    text = retry_text or text
            except asyncio.TimeoutError:
                synth_notes.append("Synthesize retry LLM call timed out (>180s).")
            except Exception as exc:
                logger.warning("Synthesize retry LLM call failed: %s", exc)
                synth_notes.append(f"Synthesize retry LLM call failed: {exc!r}")

        if report is None:
            report = _mechanical_merge(sub_results_for_llm)
            synth_notes.append(
                "Synthesize: LLM output unusable; final report assembled by mechanical merge of sub-node results."
            )
        report = report or {}
        # Coerce disease_associations[].score to float | None.
        # Some LLM outputs put strings like "N/A (Monarch entity ...)" in score;
        # normalize so the schema is well-typed.
        for d in report.get("disease_associations") or []:
            s = d.get("score")
            if s is None:
                continue
            if isinstance(s, (int, float)):
                d["score"] = float(s)
                continue
            if isinstance(s, str):
                try:
                    d["score"] = float(s.strip())
                except (ValueError, AttributeError):
                    d["score"] = None
            else:
                d["score"] = None
        # Drop pathway entries whose name AND external_id are both empty —
        # these are upstream-source error stubs (e.g. Reactome 5xx fallback).
        if isinstance(report.get("pathways"), list):
            report["pathways"] = [
                p for p in report["pathways"]
                if (p.get("name") or p.get("external_id"))
            ]
        # Always carry forward node-level notes.
        existing_notes = list(report.get("notes") or [])
        existing_notes.extend(state.get("notes") or [])
        existing_notes.extend(synth_notes)
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
        # Write final consolidated report + synthesize node log.
        run_log_dir_str = (state.get("sub_results") or {}).get("_run_log_dir")
        if run_log_dir_str:
            try:
                run_log_dir = pathlib.Path(run_log_dir_str)
                await asyncio.to_thread(run_log_dir.mkdir, parents=True, exist_ok=True)
                # Persist synthesize raw_output + parsed for post-mortem.
                await _write_node_log(
                    run_log_dir,
                    "synthesize",
                    state["target_query"],
                    synth_messages,
                    text,
                    report,
                    synth_notes,
                )
                report_text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
                await asyncio.to_thread(
                    (run_log_dir / "final_report.json").write_text, report_text
                )
                logger.info("Run log saved to %s", run_log_dir)
            except Exception as exc:
                logger.warning("Failed to write final report log: %s", exc)
        return {"final_report": report}

    graph = StateGraph(TargetDiscoveryState)
    graph.add_node("composition", composition_node)
    graph.add_node("literature", literature_node)
    graph.add_node("function", function_node)
    graph.add_node("pathway", pathway_node)
    graph.add_node("drugs", drugs_node)
    graph.add_node("synthesize", synthesize_node)

    # Topology: composition runs first to resolve UniProt + gene_symbol;
    # the four downstream nodes then fan out in parallel and all converge
    # on synthesize. This collapses worst-case wall-clock from
    # ~21 min (fully sequential) to ~max(node) ≈ 2-5 min.
    graph.add_edge(START, "composition")
    for n in ("literature", "function", "pathway", "drugs"):
        graph.add_edge("composition", n)
        graph.add_edge(n, "synthesize")
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
