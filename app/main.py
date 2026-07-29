"""
Co-Inventor — FastAPI Application Entry Point

Startup sequence:
1. Initialize SQLite database (create tables if needed)
2. Validate that at least one LLM API key is configured
3. Mount API routers
4. Serve frontend as static files at /

The app state holds:
- storage: Storage instance (shared across requests)
- event_queues: dict mapping session_id -> asyncio.Queue (for SSE streaming)
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import sessions, stream
from app.config import settings
from app.storage import Storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup + shutdown."""
    # Startup
    logger.info("Joe-Bot starting up...")

    # Warn if no LLM key is configured
    if not settings.has_llm_key:
        logger.warning(
            "No LLM API key found! Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env"
        )

    provider = "OpenRouter" if settings.use_openrouter else "Anthropic"
    model = settings.r_generation_model
    logger.info(f"LLM provider: {provider} | Default model: {model}")

    search_backends = []
    if settings.exa_api_key:
        search_backends.append("Exa")
    if settings.tavily_api_key:
        search_backends.append("Tavily")
    search_backends.append("DuckDuckGo (fallback)")
    logger.info(f"Search backends: {' → '.join(search_backends)}")

    # Initialise storage
    storage = Storage()
    await storage.init_db()
    app.state.storage = storage

    # Per-session SSE event queues
    app.state.event_queues: dict = {}

    logger.info("Joe-Bot ready ✓")

    yield

    # Shutdown
    logger.info("Joe-Bot shutting down")


app = FastAPI(
    title="Joe-Bot",
    description="AI-powered invention ideation using multi-agent Co-Scientist architecture",
    version="0.1.0",
    lifespan=lifespan,
)

# API routes
app.include_router(sessions.router, prefix="/api")
app.include_router(stream.router, prefix="/api")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "provider": "openrouter" if settings.use_openrouter else "anthropic",
        "model": settings.r_generation_model,
        "search": {
            "exa": bool(settings.exa_api_key),
            "tavily": bool(settings.tavily_api_key),
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# Serve frontend static files — must be LAST (catches all unmatched paths)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import os
    import uvicorn
    # Read PORT from environment (Railway sets this automatically)
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
