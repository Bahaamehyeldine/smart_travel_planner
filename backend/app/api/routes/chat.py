"""
chat.py
Chat endpoint — receives user query, runs LangGraph agent, returns recommendation.
Also persists agent run to database for observability.

Improvements:
- Fix 2: session_id stored in agent_runs for conversation grouping
- Fix 4: safe error messages — internal details logged, not exposed to client
"""
from datetime import datetime, timezone
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.agent.graph import build_graph
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Validated request body for POST /api/chat.
    Pydantic validates at the API boundary.
    """
    message: str
    session_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 1000:
            raise ValueError("Message too long — max 1000 characters")
        return v.strip()


class ChatResponse(BaseModel):
    """Structured response returned to the frontend."""
    response: str
    predicted_style: str
    style_confidence: float
    chunks_retrieved: int
    session_id: str | None = None


# ─────────────────────────────────────────────
# Graph singleton
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_graph():
    """
    Lazy-load the LangGraph graph.
    lru_cache ensures build_graph() runs exactly once.
    """
    logger.info("building_langgraph")
    return build_graph()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Validate request (Pydantic)
    2. Run LangGraph agent (retrieve → classify → generate)
    3. Persist agent run to database
    4. Return structured response
    """
    logger.info("chat_request", message=request.message[:50])
    started_at = datetime.now(timezone.utc)

    try:
        graph = get_graph()
        result = await graph.ainvoke({"query": request.message})

        response_text = result.get("response", "")
        predicted_style = result.get("predicted_style", "Unknown")
        style_confidence = result.get("style_confidence", 0.0)
        chunks = result.get("retrieved_chunks", [])

        # Persist to agent_runs table
        # Fix 2: session_id now stored so runs can be grouped by conversation
        # DB failure does not break the response — logged and swallowed
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    text("""
                        INSERT INTO agent_runs
                            (user_input, agent_response, status, started_at, finished_at)
                        VALUES
                            (:user_input, :agent_response, :status, :started_at, :finished_at)
                    """),
                    {
                        "user_input": request.message
                        + (f" [session:{request.session_id}]" if request.session_id else ""),
                        "agent_response": response_text,
                        "status": "completed",
                        "started_at": started_at,
                        "finished_at": datetime.now(timezone.utc),
                    }
                )
                await session.commit()
                logger.info("agent_run_saved", session_id=request.session_id)
        except Exception as db_err:
            logger.error("agent_run_save_failed", error=str(db_err))

        return ChatResponse(
            response=response_text,
            predicted_style=predicted_style,
            style_confidence=round(style_confidence, 3),
            chunks_retrieved=len(chunks),
            session_id=request.session_id,
        )

    except Exception as e:
        # Fix 4: log the real error internally, return safe message to client
        # Never expose internal stack traces or error details to the client
        logger.error("chat_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. Please try again."
        )


@router.get("/history")
async def get_history(
    limit: int = Query(default=10, ge=1, le=100)
):
    """
    Return recent agent runs for observability.
    limit is bounded between 1 and 100 — prevents abuse.
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("""
                    SELECT user_input, agent_response, status, started_at
                    FROM agent_runs
                    ORDER BY started_at DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            rows = result.fetchall()

        return {
            "runs": [
                {
                    "user_input": row[0],
                    "agent_response": (
                        row[1][:100] + "..."
                        if row[1] and len(row[1]) > 100
                        else row[1]
                    ),
                    "status": row[2],
                    "started_at": str(row[3]),
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error("history_endpoint_error", error=str(e))
        # Fix 4: safe error message
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve history. Please try again."
        )
