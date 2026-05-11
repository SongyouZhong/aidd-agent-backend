"""add projects table; add project_id/is_pinned to sessions and project_id to session_files

Revision ID: 0004_projects
Revises: 0003_session_files
Create Date: 2026-05-11
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_projects"
down_revision: Union[str, Sequence[str], None] = "0003_session_files"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. projects table
    op.create_table(
        "projects",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"])

    # 2. sessions: add nullable project_id + is_pinned
    op.add_column(
        "sessions",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "is_pinned",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # 3. session_files: add nullable project_id
    op.add_column(
        "session_files",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    # 4. Backfill: create a "Default Project" for every existing user, then point
    #    the user's sessions and session_files at it.
    op.execute(
        """
        INSERT INTO projects (id, user_id, name, description, created_at, updated_at)
        SELECT gen_random_uuid(), u.id, 'Default Project', NULL, now(), now()
        FROM users u
        WHERE NOT EXISTS (
            SELECT 1 FROM projects p WHERE p.user_id = u.id
        )
        """
    )
    op.execute(
        """
        UPDATE sessions s
        SET project_id = (
            SELECT p.id FROM projects p
            WHERE p.user_id = s.user_id
            ORDER BY p.created_at ASC
            LIMIT 1
        )
        WHERE s.project_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE session_files f
        SET project_id = (
            SELECT s.project_id FROM sessions s WHERE s.id = f.session_id
        )
        WHERE f.project_id IS NULL
        """
    )

    # 5. Tighten constraints + add FKs/indexes.
    op.alter_column("sessions", "project_id", nullable=False)
    op.create_foreign_key(
        "fk_sessions_project_id_projects",
        "sessions",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_sessions_project_id", "sessions", ["project_id"])

    op.alter_column("session_files", "project_id", nullable=False)
    op.create_foreign_key(
        "fk_session_files_project_id_projects",
        "session_files",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_session_files_project_id", "session_files", ["project_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_session_files_project_id", table_name="session_files")
    op.drop_constraint(
        "fk_session_files_project_id_projects",
        "session_files",
        type_="foreignkey",
    )
    op.drop_column("session_files", "project_id")

    op.drop_index("ix_sessions_project_id", table_name="sessions")
    op.drop_constraint(
        "fk_sessions_project_id_projects", "sessions", type_="foreignkey"
    )
    op.drop_column("sessions", "is_pinned")
    op.drop_column("sessions", "project_id")

    op.drop_index("ix_projects_user_id", table_name="projects")
    op.drop_table("projects")
