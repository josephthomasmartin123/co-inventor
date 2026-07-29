"""
Sessions API

POST /api/sessions  — create a new session and start the pipeline
GET  /api/sessions/{id}  — full session result
GET  /api/sessions        — list recent sessions
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.models.session import Session
from app.orchestrator import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateSessionRequest(BaseModel):
    problem_statement: str
    parent_session_id: str = ""     # set to start a new round from a previous session
    user_feedback: str = ""         # human steering for the next round


@router.post("/sessions")
async def create_session(body: CreateSessionRequest, request: Request):
    """
    Create a new session and start the invention pipeline as a background task.

    For a new round: pass parent_session_id (previous session) and optional user_feedback.
    The generation agent will be seeded with the prior round's top inventions + meta-review
    so it generates genuinely new ideas rather than repeating what was already found.
    """
    if not body.problem_statement.strip():
        raise HTTPException(400, "problem_statement cannot be empty")

    storage = request.app.state.storage

    # Determine round number and load prior context if this is a continuation
    prior_context = None
    round_number = 1

    if body.parent_session_id:
        parent = await storage.get_session(body.parent_session_id)
        if not parent:
            raise HTTPException(404, f"Parent session {body.parent_session_id} not found")
        if parent.status != "complete":
            raise HTTPException(400, "Parent session has not completed yet")

        round_number = parent.round_number + 1

        # Build prior context: top inventions + meta-review + user feedback
        parent_inventions = await storage.get_inventions(body.parent_session_id)
        inv_map = {i.id: i for i in parent_inventions}
        top_inventions = [
            {"title": inv_map[iid].title, "mechanism": inv_map[iid].mechanism}
            for iid in parent.final_ranked_ids if iid in inv_map
        ]

        prior_context = {
            "round_number": parent.round_number,
            "top_inventions": top_inventions,
            "meta_review": parent.meta_review or {},
            "user_feedback": body.user_feedback.strip(),
        }

    session = Session(
        problem_statement=body.problem_statement.strip(),
        round_number=round_number,
        parent_session_id=body.parent_session_id or None,
        user_feedback=body.user_feedback.strip(),
        prior_context=prior_context,
    )
    await storage.create_session(session)

    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    request.app.state.event_queues[session.id] = queue

    task = asyncio.create_task(
        run_pipeline(session, storage, queue),
        name=f"pipeline-{session.id}",
    )

    def _log_task_error(t: asyncio.Task):
        if not t.cancelled() and t.exception():
            logger.error(f"Pipeline task error: {t.exception()}")

    task.add_done_callback(_log_task_error)

    return {
        "session_id": session.id,
        "status": "pending",
        "message": "Pipeline started. Connect to /api/sessions/{id}/stream for live updates.",
    }


@router.get("/sessions")
async def list_sessions(request: Request, limit: int = 20):
    """List the most recent sessions."""
    import aiosqlite
    storage = request.app.state.storage
    async with aiosqlite.connect(storage.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, problem_statement, status, created_at, completed_at "
            "FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [
        {
            "id": row["id"],
            "problem_statement": row["problem_statement"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }
        for row in rows
    ]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """
    Return full session result including inventions, reviews, and meta-review.
    Inventions are in final_ranked Elo order.
    Reviews are keyed by invention_id for frontend lookup.
    """
    storage = request.app.state.storage
    session = await storage.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    inventions = await storage.get_inventions(session_id)
    reviews = await storage.get_reviews(session_id)

    inv_map = {i.id: i for i in inventions}

    if session.final_ranked_ids:
        ranked = [inv_map[iid] for iid in session.final_ranked_ids if iid in inv_map]
        ranked_set = set(session.final_ranked_ids)
        remaining = sorted(
            [i for i in inventions if i.id not in ranked_set],
            key=lambda x: x.elo_score, reverse=True,
        )
        ranked_inventions = ranked + remaining
    else:
        ranked_inventions = sorted(inventions, key=lambda x: x.elo_score, reverse=True)

    return {
        "session": session.model_dump(exclude={"prior_context"}),
        "inventions": [i.model_dump() for i in ranked_inventions],
        "reviews": {r.invention_id: r.model_dump() for r in reviews},
        "meta_review": session.meta_review or {},
    }
