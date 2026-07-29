"""
SSE Streaming Endpoint

GET /api/sessions/{id}/stream

Streams pipeline ProgressEvents to the browser using Server-Sent Events.
The browser connects here immediately after POST /sessions and receives
live updates as the pipeline progresses through its 5 stages.

Event types:
  status           — pipeline stage change (stage name + message)
  trigger_found    — a new invention trigger was discovered
  invention_generated — a new invention idea was created
  review_complete  — an invention was evaluated
  matchup_complete — an Elo tournament round completed
  evolution_complete — an evolved invention was created
  done             — pipeline finished; includes final_ranked_ids
  error            — pipeline failed; includes error message
  ping             — keep-alive (no data)
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.models.session import SessionStatus

logger = logging.getLogger(__name__)
router = APIRouter()

KEEPALIVE_TIMEOUT = 25.0    # Send ping every 25s to prevent proxy/browser timeout
TERMINAL_EVENTS = {"done", "error"}


@router.get("/sessions/{session_id}/stream")
async def stream_session(session_id: str, request: Request):
    """
    SSE endpoint for real-time pipeline progress.

    If the pipeline has already completed, returns the final result immediately.
    Otherwise, streams events from the pipeline's event queue.
    """
    storage = request.app.state.storage
    event_queues: dict[str, asyncio.Queue] = request.app.state.event_queues

    async def event_generator():
        # Check if already complete (e.g. client reconnecting after pipeline finished)
        session = await storage.get_session(session_id)
        if session and session.status == "complete":
            yield {
                "event": "done",
                "data": json.dumps({
                    "message": "already_complete",
                    "final_ranked_ids": session.final_ranked_ids,
                    "session_id": session_id,
                }),
            }
            return
        if session and session.status == "failed":
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": session.error or "Pipeline failed",
                    "session_id": session_id,
                }),
            }
            return

        queue = event_queues.get(session_id)
        if queue is None:
            # Queue not found but session not complete — race condition at startup
            # Wait briefly for the pipeline task to initialise
            for _ in range(5):
                await asyncio.sleep(0.5)
                queue = event_queues.get(session_id)
                if queue is not None:
                    break

        if queue is None:
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": "Session stream not found. Try GET /api/sessions/{id} for results.",
                    "session_id": session_id,
                }),
            }
            return

        try:
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    logger.debug(f"Client disconnected from stream {session_id}")
                    break

                try:
                    event_data = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_TIMEOUT)
                except asyncio.TimeoutError:
                    # Send keep-alive ping to prevent timeout
                    yield {"event": "ping", "data": json.dumps({"session_id": session_id})}
                    continue

                event_type = event_data.get("event", "unknown")
                yield {
                    "event": event_type,
                    "data": json.dumps(event_data),
                }

                if event_type in TERMINAL_EVENTS:
                    # Clean up the queue — pipeline is done
                    event_queues.pop(session_id, None)
                    break

        except asyncio.CancelledError:
            logger.debug(f"SSE stream cancelled for {session_id}")
        except Exception as e:
            logger.error(f"SSE stream error for {session_id}: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e), "session_id": session_id}),
            }

    return EventSourceResponse(event_generator())
