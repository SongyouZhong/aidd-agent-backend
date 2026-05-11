"""Deep target-discovery tool.

Wraps the heavy 5-node ``target_discovery_graph`` as a normal LangChain
``@tool`` so the main ReAct loop in ``chat_service.stream_chat`` can
trigger it via native function-calling. This is the "Agent-as-a-Tool"
architecture: the conversation drives the graph, not the other way
round.

Behaviour:
    1. Stream the sub-graph with ``astream`` and forward node-transition
       events through the per-request ``progress_callback`` contextvar
       (consumed by ``chat_service`` to emit ``research_progress`` SSE).
    2. Persist the full ``TargetReport`` JSON as a SessionFile (so the
       frontend can offer a download and we avoid bloating future LLM
       turns with a 5-15k-token tool message).
    3. Return a compact summary dict (JSON-stringified) to the LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.agent.llm_provider import get_default_provider
from app.services import target_report_service
from app.services.chat_context import (
    deep_research_running,
    get_chat_context,
    get_progress_callback,
)

logger = logging.getLogger(__name__)

# Global wall-clock cap. The sub-graph itself enforces per-node budgets
# (composition 240s, others 180-200s) so this is a safety net for the
# whole pipeline.
DEEP_RESEARCH_TIMEOUT_SECONDS = 600.0


def _build_summary(target_query: str, report: dict[str, Any]) -> dict[str, Any]:
    """Compact summary fed back to the orchestrator LLM."""
    fn = (report.get("function_narrative") or "").strip()
    return {
        "target": target_query,
        "target_meta": report.get("target") or {},
        "counts": {
            "proteins": len(report.get("proteins") or []),
            "papers": len(report.get("papers") or []),
            "disease_associations": len(report.get("disease_associations") or []),
            "pathways": len(report.get("pathways") or []),
            "small_molecule_drugs": len(report.get("small_molecule_drugs") or []),
            "peptide_drugs": len(report.get("peptide_drugs") or []),
            "antibody_drugs": len(report.get("antibody_drugs") or []),
        },
        "function_brief": fn[:400] + ("..." if len(fn) > 400 else ""),
        "notes": report.get("notes") or [],
    }


async def _run_pipeline(target_query: str) -> dict[str, Any]:
    """Drive the sub-graph with progress streaming."""
    # Lazy import: target_discovery_graph imports default_registry from
    # app.tools, which during package init would re-enter this module.
    from app.agent.target_discovery_graph import build_target_discovery_graph

    provider = get_default_provider()
    td_graph = build_target_discovery_graph(provider)
    progress = get_progress_callback()

    initial = {
        "target_query": target_query,
        "messages": [],
        "sub_results": {},
        "notes": [],
        "final_report": {},
    }

    final_report: dict[str, Any] = {}
    seen_phases: set[str] = set()

    async for event in td_graph.astream(initial):
        # ``astream`` yields dicts keyed by the node name that just
        # finished (LangGraph default streaming mode).
        if not isinstance(event, dict):
            continue
        for node_name, payload in event.items():
            if node_name in seen_phases:
                continue
            seen_phases.add(node_name)
            if progress is not None:
                try:
                    await progress(
                        "research_progress",
                        {"phase": node_name, "target": target_query},
                    )
                except Exception:
                    logger.warning("progress callback failed", exc_info=True)
            # Capture the final report when the synthesize node fires.
            if isinstance(payload, dict) and payload.get("final_report"):
                final_report = payload["final_report"]
    return final_report or {}


@tool
async def run_target_discovery(target_query: str) -> str:
    """Run a multi-source deep research pipeline for a protein/gene target.

    Use this whenever the user asks to analyze, research, profile, or
    "discover" a biological target (gene symbol or protein name, e.g.
    "TARDBP", "TDP-43", "EGFR"). The pipeline gathers literature,
    structure/function, pathway, and drug evidence in parallel and
    takes about 2-5 minutes.

    The full TargetReport JSON is persisted as a downloadable session
    file. This tool returns a compact JSON summary containing the
    counts per evidence dimension, a brief functional narrative, and
    the ``report_file_id`` you can reference in your reply so the user
    can download the full report.

    Args:
        target_query: Gene symbol or protein name (e.g. "TARDBP").
    """
    # Re-entrancy guard.
    if deep_research_running.get():
        return json.dumps(
            {
                "status": "error",
                "message": (
                    "Another target-discovery pipeline is already running in "
                    "this turn. Wait for it to finish before calling this tool again."
                ),
            },
            ensure_ascii=False,
        )

    ctx = get_chat_context()
    if ctx is None:
        return json.dumps(
            {
                "status": "error",
                "message": "run_target_discovery requires a chat session context.",
            },
            ensure_ascii=False,
        )

    token = deep_research_running.set(True)
    try:
        try:
            report = await asyncio.wait_for(
                _run_pipeline(target_query),
                timeout=DEEP_RESEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"Target discovery exceeded {DEEP_RESEARCH_TIMEOUT_SECONDS:.0f}s "
                        "wall-clock budget."
                    ),
                    "target": target_query,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.exception("Deep-research pipeline crashed")
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Pipeline crashed: {exc!r}",
                    "target": target_query,
                },
                ensure_ascii=False,
            )

        if not report:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Pipeline returned an empty report.",
                    "target": target_query,
                },
                ensure_ascii=False,
            )

        # Persist as session file.
        file_id: str | None = None
        filename: str | None = None
        try:
            record = await target_report_service.save_report_as_session_file(
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                project_id=ctx.project_id,
                target_query=target_query,
                report=report,
            )
            file_id = str(record.id)
            filename = record.original_filename
        except Exception as exc:
            logger.exception("Failed to persist target report as session file")
            # Still return the summary to the LLM, but flag the failure.
            summary = _build_summary(target_query, report)
            summary["status"] = "ok_no_file"
            summary["persist_error"] = repr(exc)
            return json.dumps(summary, ensure_ascii=False)

        summary = _build_summary(target_query, report)
        summary["status"] = "ok"
        summary["report_file_id"] = file_id
        summary["report_filename"] = filename
        return json.dumps(summary, ensure_ascii=False)
    finally:
        deep_research_running.reset(token)
