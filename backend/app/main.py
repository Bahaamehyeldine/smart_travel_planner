"""
main.py
FastAPI application entry point.

Run from backend/ directory:
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router

logger = structlog.get_logger(__name__)


# Fix 2: lifespan context manager replaces deprecated
# @app.on_event("startup") and @app.on_event("shutdown")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Code before yield runs on startup.
    Code after yield runs on shutdown.
    Replaces deprecated @app.on_event decorators.
    """
    logger.info("api_startup", version="1.0.0")
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="Smart Travel Planner API",
    description="AI-powered travel recommendations using RAG + ML classification",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React frontend on localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
