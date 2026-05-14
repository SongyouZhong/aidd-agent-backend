"""Deep target-discovery tool.

Wraps the heavy 5-node ``target_discovery_graph`` as a normal LangChain
``@tool`` so the main ReAct loop in ``chat_service.stream_chat`` can
trigger it via native function-calling. This is the "Agent-as-a-Tool"
architecture: the conversation drives the graph, not the other way
round.

Behaviour (post-decoupling):
    1. ``run_target_discovery`` submits the pipeline as a background task
       and returns ``{"status": "accepted", "task_id": ...}`` immediately
       so the main chat SSE finishes in seconds.
    2. ``_run_discovery_background`` runs the 5-node graph, reporting
       progress via Redis Pub/Sub (task_registry).
    3. On completion the background task calls
       ``chat_service.resume_after_task`` to persist the final assistant
       summary message and broadcast a ``message_appended`` event.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.services import target_report_service
from app.services.background_runner import background_runner
from app.services import task_registry
from app.services.chat_context import (
    current_chat_context,
    deep_research_running,
    get_chat_context,
    progress_callback,
)

logger = logging.getLogger(__name__)

# Global wall-clock cap per background task.
DEEP_RESEARCH_TIMEOUT_SECONDS = 900.0

# Progress percent and description per pipeline node completion.
_PHASE_PERCENT: dict[str, int] = {
    "composition": 15,
    "literature":  35,
    "function":    50,
    "pathway":     65,
    "drugs":       80,
    "synthesize":  95,
}
_PHASE_DESC: dict[str, str] = {
    "composition": "Resolving target structure...",
    "literature":  "Analyzing literature...",
    "function":    "Analyzing target function...",
    "pathway":     "Analyzing pathway associations...",
    "drugs":       "Analyzing drug interactions...",
    "synthesize":  "Synthesizing results...",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_summary(target_query: str, report: dict[str, Any]) -> dict[str, Any]:
    """Compact summary stored in task result and fed back to the LLM."""
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


async def _run_pipeline(target_query: str, language: str = "English") -> dict[str, Any]:
    """Drive the sub-graph, forwarding node events through progress_callback."""
    # Lazy import avoids a circular import: target_discovery_graph imports
    # default_registry from app.tools which would re-enter this module.
    from app.agent.target_discovery_graph import build_target_discovery_graph
    from app.agent.llm_provider import get_default_provider
    from app.services.chat_context import get_progress_callback

    provider = get_default_provider()
    td_graph = build_target_discovery_graph(provider)
    progress = get_progress_callback()

    initial = {
        "target_query": target_query,
        "language": language,
        "messages": [],
        "sub_results": {},
        "notes": [],
        "final_report": {},
    }

    final_report: dict[str, Any] = {}
    seen_phases: set[str] = set()

    async for event in td_graph.astream(initial):
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
            if isinstance(payload, dict) and payload.get("final_report"):
                final_report = payload["final_report"]
    return final_report or {}


async def _run_discovery_background(
    task_id: str,
    target_query: str,
    session_id: str,
    user_id: str,
    project_id: str | None,
    language: str = "English",
) -> None:
    """Background coroutine: runs the full pipeline and finalises the task.

    Called via ``background_runner.submit``.  Overrides the progress callback
    contextvar so progress events go to Redis rather than the (already-closed)
    SSE progress queue.
    """
    # Override progress_callback for this background task's context so
    # _run_pipeline publishes to Redis Pub/Sub instead of the SSE queue.
    async def _bg_progress(event_type: str, payload: dict[str, Any]) -> None:
        phase = payload.get("phase", "")
        percent = _PHASE_PERCENT.get(phase, 0)
        desc = _PHASE_DESC.get(phase, phase)
        await task_registry.update_progress(task_id, percent, phase, desc)

    progress_callback.set(_bg_progress)

    # Mark as running immediately.
    await task_registry.update_progress(task_id, 5, "composition", "Resolving target structure...")

    try:
        try:
            report = await asyncio.wait_for(
                _run_pipeline(target_query, language=language),
                timeout=DEEP_RESEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await task_registry.fail(task_id, f"timeout after {DEEP_RESEARCH_TIMEOUT_SECONDS:.0f}s")
            return
        except asyncio.CancelledError:
            await task_registry.cancel(task_id)
            raise
        except Exception as exc:
            logger.exception("Background discovery pipeline crashed for task %s", task_id)
            await task_registry.fail(task_id, repr(exc))
            return

        if not report:
            await task_registry.fail(task_id, "pipeline returned an empty report")
            return

        # Persist the report as session files.
        try:
            saved = await target_report_service.save_report_as_session_file(
                session_id=session_id,
                user_id=user_id,
                project_id=project_id,
                target_query=target_query,
                report=report,
                language=language,
            )
        except Exception as exc:
            logger.exception("Failed to persist target report for task %s", task_id)
            await task_registry.fail(task_id, f"persist error: {exc!r}")
            return

        summary = _build_summary(target_query, report)
        file_ids = [str(saved.md_record.id), str(saved.json_record.id)]
        result = {
            **summary,
            "file_ids": file_ids,
            "report_md_file_id": str(saved.md_record.id),
            "report_md_filename": saved.md_record.original_filename,
            "report_file_id": str(saved.json_record.id),
            "report_filename": saved.json_record.original_filename,
        }
        await task_registry.complete(task_id, result)

        # Trigger auto-resume: persist assistant summary + broadcast message_appended.
        # Lazy import to avoid circular dependency (chat_service → tools → deep_research).
        from app.services.chat_service import resume_after_task
        await resume_after_task(session_id, task_id)

    except asyncio.CancelledError:
        # Already handled above; re-raise so asyncio knows the task ended.
        raise
    except Exception as exc:
        logger.exception("Unexpected error in background discovery task %s", task_id)
        try:
            await task_registry.fail(task_id, repr(exc))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LangChain @tool — called by the main ReAct loop
# ---------------------------------------------------------------------------

@tool
async def run_target_discovery(target_query: str) -> str:
    """Run a multi-source deep research pipeline for a protein/gene target.

    Use this whenever the user asks to analyze, research, profile, or
    "discover" a biological target (gene symbol or protein name, e.g.
    "TARDBP", "TDP-43", "EGFR"). The pipeline gathers literature,
    structure/function, pathway, and drug evidence in parallel and
    takes about 2-5 minutes.

    This tool returns immediately with an "accepted" status — the pipeline
    runs as a background task. Progress is streamed through the dedicated
    /sessions/{id}/events SSE channel and the final report appears in the
    right-side panel when done.

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
                    "this session. Wait for it to finish before launching another."
                ),
            },
            ensure_ascii=False,
        )

    ctx = get_chat_context()
    if ctx is None:
        return json.dumps(
            {"status": "error", "message": "run_target_discovery requires a chat session context."},
            ensure_ascii=False,
        )

    task_id = await task_registry.create_task(
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        project_id=ctx.project_id,
        kind="target_discovery",
        target=target_query,
    )

    await background_runner.submit(
        _run_discovery_background(
            task_id=task_id,
            target_query=target_query,
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            project_id=ctx.project_id,
            language=ctx.language,
        ),
        task_id=task_id,
    )

    # Mark re-entrancy so the LLM can't launch a second pipeline this turn.
    deep_research_running.set(True)

    return json.dumps(
        {
            "status": "accepted",
            "task_id": task_id,
            "target": target_query,
            "eta_seconds": 300,
            "message": (
                "Target discovery task accepted. Running in the background — "
                "results will be appended to this conversation when ready."
            ),
        },
        ensure_ascii=False,
    )
