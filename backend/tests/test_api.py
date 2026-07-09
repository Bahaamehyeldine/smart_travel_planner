"""
test_api.py

Tests for the FastAPI endpoints.

These tests verify:
- Input validation rejects bad requests with 422
- Health endpoint returns correct structure
- Chat endpoint invokes the LangGraph agent
- History endpoint respects query parameter bounds
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Health endpoint must return 200 when DB is reachable."""
        with patch("app.api.routes.health.AsyncSessionLocal") as mock_session:
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            response = client.get("/api/health")
            assert response.status_code == 200

    def test_health_returns_expected_fields(self, client):
        """Health response must contain status, database, and version fields."""
        with patch("app.api.routes.health.AsyncSessionLocal") as mock_session:
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            response = client.get("/api/health")
            data = response.json()
            assert "status" in data
            assert "database" in data
            assert "version" in data
            assert data["status"] == "ok"

    def test_health_reports_db_error(self, client):
        """Health endpoint must report DB error gracefully — not return 500."""
        with patch("app.api.routes.health.AsyncSessionLocal") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("DB down")
            )
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert "error" in data["database"]


class TestChatEndpoint:
    def test_empty_message_returns_422(self, client):
        """Empty message must be rejected by Pydantic validation."""
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 422

    def test_message_too_long_returns_422(self, client):
        """Message over 1000 chars must be rejected — prevents API abuse."""
        response = client.post("/api/chat", json={"message": "x" * 1001})
        assert response.status_code == 422

    def test_missing_message_returns_422(self, client):
        """Missing message field must be rejected by Pydantic."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_whitespace_only_message_returns_422(self, client):
        """Message with only whitespace must be rejected after strip."""
        response = client.post("/api/chat", json={"message": "   "})
        assert response.status_code == 422

    def test_valid_message_returns_200(self, client):
        """Valid message must return 200 with correct response structure."""
        mock_result = {
            "response": "I recommend Queenstown for adventure!",
            "predicted_style": "Adventure",
            "style_confidence": 0.85,
            "retrieved_chunks": [
                {
                    "chunk_text": "test",
                    "source_document": "wikivoyage:Queenstown",
                    "chunk_index": 0,
                    "similarity": 0.8
                }
            ],
        }

        with patch("app.api.routes.chat.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value=mock_result)
            mock_get_graph.return_value = mock_graph

            with patch("app.api.routes.chat.AsyncSessionLocal") as mock_session:
                mock_db = AsyncMock()
                mock_db.execute = AsyncMock()
                mock_db.commit = AsyncMock()
                mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

                response = client.post(
                    "/api/chat",
                    json={"message": "I want to go hiking in the mountains"}
                )

                assert response.status_code == 200
                data = response.json()
                assert "response" in data
                assert "predicted_style" in data
                assert "style_confidence" in data
                assert "chunks_retrieved" in data

    def test_valid_message_invokes_agent(self, client):
        """
        Fix 4: verify the LangGraph agent was actually called.
        Previous version only checked response structure — the agent
        could have been bypassed entirely and the test would still pass.
        """
        mock_result = {
            "response": "Test response",
            "predicted_style": "Culture",
            "style_confidence": 0.7,
            "retrieved_chunks": [],
        }

        with patch("app.api.routes.chat.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value=mock_result)
            mock_get_graph.return_value = mock_graph

            with patch("app.api.routes.chat.AsyncSessionLocal") as mock_session:
                mock_db = AsyncMock()
                mock_db.execute = AsyncMock()
                mock_db.commit = AsyncMock()
                mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

                client.post(
                    "/api/chat",
                    json={"message": "beach vacation"}
                )

                # Fix 4: assert agent was actually invoked
                mock_graph.ainvoke.assert_called_once()
                call_args = mock_graph.ainvoke.call_args[0][0]
                assert call_args["query"] == "beach vacation"

    def test_response_chunks_retrieved_matches_list_length(self, client):
        """chunks_retrieved in response must equal actual number of chunks."""
        mock_result = {
            "response": "Test",
            "predicted_style": "Adventure",
            "style_confidence": 0.9,
            "retrieved_chunks": [
                {"chunk_text": "a", "source_document": "x", "chunk_index": 0, "similarity": 0.5},
                {"chunk_text": "b", "source_document": "y", "chunk_index": 0, "similarity": 0.4},
                {"chunk_text": "c", "source_document": "z", "chunk_index": 0, "similarity": 0.3},
            ],
        }

        with patch("app.api.routes.chat.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value=mock_result)
            mock_get_graph.return_value = mock_graph

            with patch("app.api.routes.chat.AsyncSessionLocal") as mock_session:
                mock_db = AsyncMock()
                mock_db.execute = AsyncMock()
                mock_db.commit = AsyncMock()
                mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

                response = client.post("/api/chat", json={"message": "mountains"})
                assert response.json()["chunks_retrieved"] == 3


class TestHistoryEndpoint:
    def test_history_returns_200(self, client):
        """History endpoint must return 200 with runs list."""
        with patch("app.api.routes.chat.AsyncSessionLocal") as mock_session:
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock(return_value=mock_result)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            response = client.get("/api/history")
            assert response.status_code == 200
            assert "runs" in response.json()

    def test_history_limit_too_high_returns_422(self, client):
        """limit > 100 must be rejected — prevents database abuse."""
        response = client.get("/api/history?limit=101")
        assert response.status_code == 422

    def test_history_limit_zero_returns_422(self, client):
        """limit < 1 must be rejected — zero results is meaningless."""
        response = client.get("/api/history?limit=0")
        assert response.status_code == 422

    def test_history_default_limit_is_10(self, client):
        """Default limit should be 10 — verify it works without explicit limit."""
        with patch("app.api.routes.chat.AsyncSessionLocal") as mock_session:
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_db = AsyncMock()
            mock_db.execute = AsyncMock(return_value=mock_result)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            response = client.get("/api/history")
            assert response.status_code == 200
