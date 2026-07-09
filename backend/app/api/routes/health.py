"""
health.py
Health check endpoint — verifies API and database are reachable.
"""
from fastapi import APIRouter
from sqlalchemy import text
from app.core.database import AsyncSessionLocal
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns API status and database connectivity.
    Used by Docker Compose and monitoring tools.
    """
    db_status = "ok"
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error("db_health_check_failed", error=str(e))

    return {
        "status": "ok",
        "database": db_status,
        "version": "1.0.0",
    }
