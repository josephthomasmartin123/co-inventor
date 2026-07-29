"""Entry point — delegates to app/main.py."""
from app.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
