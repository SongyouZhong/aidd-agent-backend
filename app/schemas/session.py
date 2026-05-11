"""Session API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    is_pinned: bool | None = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> "SessionUpdate":
        if self.title is None and self.is_pinned is None:
            raise ValueError("At least one of 'title' or 'is_pinned' must be provided")
        return self


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    is_pinned: bool
    created_at: datetime
    updated_at: datetime
