"""
api/main.py — FastAPI application factory.

START THE SERVER WITH:
    python run.py              ← recommended (no warnings)
    uvicorn api.main:app --reload --port 8000  ← also works
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import circuits, devices, jobs
from config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Quantum ML API starting on port %d", settings.api_port)
    logger.info("IBM token: %s", "configured" if settings.ibm_quantum_token else "not set (simulator modes only)")
    yield
    logger.info("Quantum ML API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quantum ML API",
        description=(
            "Hybrid quantum-classical ML pipeline.\n\n"
            "**Modes:** `simulator` · `noisy_simulator` · `real` (IBM token required)\n\n"
            "No IBM account needed for the first two modes."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "error_type": type(exc).__name__},
        )

    app.include_router(circuits.router)
    app.include_router(jobs.router)
    app.include_router(devices.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Quantum ML API", "version": "1.0.0", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
