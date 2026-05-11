"""Trace API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TraceStepResponse(BaseModel):
    step_number: int
    step_type: str  # "think" | "act" | "observe" | "compacted"
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result_summary: str | None = None
    raw_data_uri: str | None = None
    latency_ms: int | None = None
    created_at: str | None = None

    class Config:
        extra = "ignore"  # Tolerate extra fields from stored JSONL

