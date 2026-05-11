"""SQLAlchemy ORM models.

All model modules MUST be imported here so that Alembic's autogenerate
can discover them via ``Base.metadata``.
"""

from app.models.user import User  # noqa: F401
from app.models.project import Project  # noqa: F401
from app.models.session import Session  # noqa: F401
from app.models.target import ProteinRecord, Target  # noqa: F401
from app.models.pathway_drug import (  # noqa: F401
    DiseaseAssociation,
    Drug,
    Paper,
    Pathway,
    TargetDrugActivity,
    target_paper,
    target_pathway,
)
from app.models.target_report import TargetReport  # noqa: F401
from app.models.session_file import SessionFile  # noqa: F401

__all__ = [
    "User",
    "Project",
    "Session",
    "Target",
    "ProteinRecord",
    "Pathway",
    "Drug",
    "TargetDrugActivity",
    "DiseaseAssociation",
    "Paper",
    "TargetReport",
    "SessionFile",
]
