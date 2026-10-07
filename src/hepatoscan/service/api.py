"""FastAPI app (extra ``api``). All logic is in ``handlers.py``; this module only maps HTTP to it."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .handlers import AuthError, NotConfigured, Service
from .validation import UploadError

STATIC = Path(__file__).with_name("static")


def create_app(service: Service) -> FastAPI:
    app = FastAPI(title="hepatoscan", docs_url=None, redoc_url=None, openapi_url=None)

    def _errors(fn):
        try:
            return fn()
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc
        except NotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        except (UploadError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:"
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": service.segmenter.name, "llm": service.assistant.llm.name}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/app.js")
    def script():
        return HTMLResponse((STATIC / "app.js").read_text(encoding="utf-8"), media_type="application/javascript")

    @app.post("/segment")
    async def segment(request: Request, authorization: str | None = Header(default=None),
                      x_filename: str | None = Header(default=None)) -> JSONResponse:
        length = int(request.headers.get("content-length") or 0)
        if length > service.max_upload_bytes:
            raise HTTPException(413, "upload too large")
        body = await request.body()
        return JSONResponse(_errors(lambda: service.segment_upload(body, x_filename, authorization)))

    @app.post("/chat")
    async def chat(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(422, "body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(422, "body must be a JSON object")
        return JSONResponse(_errors(lambda: service.chat(payload, authorization)))

    @app.delete("/chat/{session_id}")
    def end_chat(session_id: str, authorization: str | None = Header(default=None)) -> dict:
        return _errors(lambda: service.end_chat(session_id, authorization))

    return app
