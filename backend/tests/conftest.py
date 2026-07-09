"""
conftest.py

Shared pytest fixtures available across all test files.
Centralizing fixtures here avoids duplication and makes
the test suite easier to maintain.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    Create a FastAPI TestClient for endpoint testing.
    Shared across test_api.py and any future API test files.
    TestClient runs the ASGI app synchronously — no live server needed.
    """
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """
    Reusable mock for AsyncSessionLocal.
    Prevents real database calls during unit tests.
    """
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()

    with patch("app.core.database.AsyncSessionLocal") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        yield mock_db
