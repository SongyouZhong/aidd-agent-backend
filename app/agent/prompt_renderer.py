"""Jinja2-based Prompt renderer (design doc §8.2).

Renders the System Prompt with dynamic slots:
  * ``current_time``        — system clock at call time
  * ``active_tools``        — names exposed to the LLM this turn
  * ``hot_loaded_hint``     — tools just mounted via tool_search
  * ``session_memory``      — long-term compressed memory (Phase 5 fills this)
  * ``system_status``       — free-form status flag (e.g. "plan_mode")
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from jinja2 import Environment, StrictUndefined

from app.agent.prompts.templates import SYSTEM_PROMPT_TEMPLATE

_env = Environment(undefined=StrictUndefined, autoescape=False, trim_blocks=False)
_system_template = _env.from_string(SYSTEM_PROMPT_TEMPLATE)


def render_system_prompt(
    *,
    active_tools: Iterable[str],
    hot_loaded: Iterable[str] | None = None,
    session_memory: str | None = None,
    system_status: str = "ready",
    now: datetime | None = None,
) -> str:
    return _system_template.render(
        current_time=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        active_tools=list(active_tools),
        hot_loaded_hint=", ".join(hot_loaded) if hot_loaded else "",
        session_memory=session_memory or "",
        system_status=system_status,
    )
