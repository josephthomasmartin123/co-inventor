from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

SessionStatus = Literal[
    "pending",
    "generating",
    "proximity",
    "reflecting",
    "ranking",
    "evolving",
    "meta_reviewing",
    "complete",
    "failed",
]


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    problem_statement: str
    status: SessionStatus = "pending"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    final_ranked_ids: list[str] = []
    meta_review: Optional[dict] = None

    # Multi-round fields
    round_number: int = 1                           # 1-indexed
    parent_session_id: Optional[str] = None         # previous round's session id
    user_feedback: str = ""                         # steering note from user between rounds

    # Runtime only (not persisted) — loaded by the API before pipeline starts
    prior_context: Optional[dict] = None


class ProgressEvent(BaseModel):
    """Emitted over SSE during pipeline execution."""
    event: str
    data: dict
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
