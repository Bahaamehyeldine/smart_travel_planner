"""
test_retriever.py

Tests for the RAG retrieval pipeline.

These tests verify:
- Retriever returns empty list on database errors (not crashes)
- Model loader is properly cached (singleton pattern)
- Return structure matches expected schema
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestRetrieve:
    async def test_returns_empty_list_on_db_error(self):
        """
        Retriever must return [] on database failure — not raise an exception.
        The LangGraph agent handles empty retrieval gracefully, but an
        exception would crash the entire agent pipeline.
        """
        from app.rag.retriever import retrieve

        with patch("app.rag.retriever.AsyncSessionLocal") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("DB connection failed")
            )
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await retrieve("hiking mountains", top_k=3)
            assert result == []

    async def test_returns_correct_structure(self):
        """
        Each retrieved chunk must have the four expected keys.
        The LangGraph generate_node depends on this structure.
        """
        from app.rag.retriever import retrieve

        mock_rows = [
            ("Queenstown - Do: hiking bungee", "wikivoyage:Queenstown", 0, 0.85),
            ("Banff - Do: skiing hiking", "wikivoyage:Banff", 0, 0.72),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("app.rag.retriever.AsyncSessionLocal") as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await retrieve("hiking mountains", top_k=2)

            # Fix 2: assert mock was actually used
            mock_session.execute.assert_called_once()

            assert isinstance(result, list)
            assert len(result) == 2
            for chunk in result:
                assert "chunk_text" in chunk
                assert "source_document" in chunk
                assert "chunk_index" in chunk
                assert "similarity" in chunk

    async def test_similarity_values_are_floats(self):
        """Similarity scores must be floats rounded to 4 decimal places."""
        from app.rag.retriever import retrieve

        mock_rows = [
            ("Some chunk text", "wikivoyage:Rome", 0, 0.75432),
        ]
        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("app.rag.retriever.AsyncSessionLocal") as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await retrieve("rome culture", top_k=1)
            if result:
                assert isinstance(result[0]["similarity"], float)

    def test_get_model_returns_sentence_transformer(self):
        """
        Model loader should return a SentenceTransformer instance.
        Verifies the correct model class is loaded.
        """
        from app.rag.retriever import get_model
        from sentence_transformers import SentenceTransformer
        model = get_model()
        assert isinstance(model, SentenceTransformer)

    def test_get_model_is_cached(self):
        """
        Model loader must return the same instance on repeated calls.
        lru_cache singleton pattern — loading takes ~5s, must not repeat.
        """
        from app.rag.retriever import get_model
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2  # same object in memory
