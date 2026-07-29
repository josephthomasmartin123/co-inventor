"""
Storage layer — SQLite via aiosqlite.

Tables:
  sessions   — session metadata, status, final_ranked_ids, meta_review JSON
  inventions — invention objects (JSON blob per row)
  reviews    — review objects (JSON blob per row)

Everything is stored as JSON blobs — no column-per-field mapping.
Simple to evolve during development; no migrations needed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import aiosqlite

from app.config import settings
from app.models.invention import Invention, Review
from app.models.session import Session, SessionStatus

logger = logging.getLogger(__name__)

CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    problem_statement TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT,
    completed_at TEXT,
    error TEXT,
    final_ranked_ids TEXT NOT NULL DEFAULT '[]',
    meta_review TEXT DEFAULT '{}',
    round_number INTEGER NOT NULL DEFAULT 1,
    parent_session_id TEXT,
    user_feedback TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS inventions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    invention_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inventions_session ON inventions(session_id);
CREATE INDEX IF NOT EXISTS idx_reviews_session ON reviews(session_id);
"""


class Storage:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_path

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_SCHEMA)
            await db.commit()
        logger.info(f"Database ready: {self.db_path}")

    # ── Sessions ──────────────────────────────────────────────────────────

    async def create_session(self, session: Session) -> Session:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO sessions
                   (id, problem_statement, status, created_at, final_ranked_ids, meta_review,
                    round_number, parent_session_id, user_feedback)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session.id, session.problem_statement, session.status,
                 session.created_at, "[]", "{}",
                 session.round_number, session.parent_session_id, session.user_feedback),
            )
            await db.commit()
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            return None
        return Session(
            id=row["id"],
            problem_statement=row["problem_statement"],
            status=row["status"],
            created_at=row["created_at"] or "",
            completed_at=row["completed_at"],
            error=row["error"],
            final_ranked_ids=json.loads(row["final_ranked_ids"] or "[]"),
            meta_review=json.loads(row["meta_review"] or "{}") or None,
            round_number=row["round_number"] or 1,
            parent_session_id=row["parent_session_id"],
            user_feedback=row["user_feedback"] or "",
        )

    async def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        error: Optional[str] = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE sessions SET status = ?, error = ? WHERE id = ?",
                (status, error, session_id),
            )
            await db.commit()

    async def finalize_session(
        self,
        session_id: str,
        final_ranked_ids: list[str],
        meta_review: Optional[dict] = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE sessions
                   SET status = 'complete', completed_at = ?,
                       final_ranked_ids = ?, meta_review = ?
                   WHERE id = ?""",
                (
                    datetime.utcnow().isoformat(),
                    json.dumps(final_ranked_ids),
                    json.dumps(meta_review or {}),
                    session_id,
                ),
            )
            await db.commit()

    # ── Inventions ────────────────────────────────────────────────────────

    async def save_inventions(self, inventions: list[Invention]) -> None:
        if not inventions:
            return
        async with aiosqlite.connect(self.db_path) as db:
            for inv in inventions:
                await db.execute(
                    "INSERT OR REPLACE INTO inventions (id, session_id, data) VALUES (?, ?, ?)",
                    (inv.id, inv.session_id, inv.model_dump_json()),
                )
            await db.commit()

    async def update_inventions(self, inventions: list[Invention]) -> None:
        await self.save_inventions(inventions)

    async def get_inventions(self, session_id: str) -> list[Invention]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM inventions WHERE session_id = ?", (session_id,)
            ) as cursor:
                rows = await cursor.fetchall()
        return [Invention.model_validate_json(row[0]) for row in rows]

    # ── Reviews ───────────────────────────────────────────────────────────

    async def save_reviews(self, reviews: list[Review]) -> None:
        if not reviews:
            return
        async with aiosqlite.connect(self.db_path) as db:
            for r in reviews:
                await db.execute(
                    """INSERT OR REPLACE INTO reviews
                       (id, invention_id, session_id, data) VALUES (?, ?, ?, ?)""",
                    (r.id, r.invention_id, r.session_id, r.model_dump_json()),
                )
            await db.commit()

    async def get_reviews(self, session_id: str) -> list[Review]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM reviews WHERE session_id = ?", (session_id,)
            ) as cursor:
                rows = await cursor.fetchall()
        return [Review.model_validate_json(row[0]) for row in rows]
