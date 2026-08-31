from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import make_router
from .config import Settings
from .resources import ResourceConflict
from .service import InvestRAGService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="InvestRAG Studio API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["*"], allow_headers=["*"])
    service = InvestRAGService(settings)
    app.state.service = service
    app.include_router(make_router(service))

    @app.exception_handler(ResourceConflict)
    def resource_conflict_handler(_request: Request, exc: ResourceConflict) -> JSONResponse:
        # design §17: a resource conflict is a visible, typed refusal, never a crash.
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "component": exc.component.value,
                "active_profile": exc.profile.value,
            },
        )

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    return app


app = create_app()
