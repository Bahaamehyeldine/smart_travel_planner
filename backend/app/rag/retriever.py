"""
retriever.py

Retrieves relevant destination chunks from pgvector given a user query.

The retrieval step in the RAG pipeline:
1. Embed the user query with the same model used for indexing
2. Run cosine similarity search against pgvector
3. Return top-k most relevant chunks

Critical: must use the same model as indexer.py (all-MiniLM-L6-v2)
If models differ, vector spaces are incompatible and results are meaningless.
"""

import asyncio
from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from app.core.database import AsyncSessionLocal

logger = structlog.get_logger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

# Default minimum similarity threshold
# 0.3 filters out genuinely irrelevant chunks (orthogonal vectors score ~0.0)
# Adjust based on observed retrieval quality
DEFAULT_MIN_SIMILARITY = 0.3


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model — cached after first call.

    Fix 2: uses lru_cache instead of mutable global variable.
    lru_cache(maxsize=1) ensures the model is loaded exactly once
    and reused across all calls — same behavior as a singleton
    but without the mutable global anti-pattern.

    Loading SentenceTransformer takes ~5 seconds — we don't want
    that overhead on every query.
    """
    logger.info("loading_embedding_model", model=MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


async def retrieve(
    query: str,
    top_k: int = 5,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[dict]:
    """
    Retrieve top-k most relevant chunks for a given query.

    Args:
        query: natural language query from the user
               e.g. "beaches with good snorkeling in Southeast Asia"
        top_k: number of chunks to return (default 5)
        min_similarity: minimum cosine similarity threshold
                        default 0.3 — filters out irrelevant chunks
                        cosine similarity: 0.0=orthogonal, 1.0=identical

    Returns:
        list of dicts with keys:
            - chunk_text: the retrieved text
            - source_document: e.g. "wikivoyage:Queenstown"
            - chunk_index: section index within the article
            - similarity: cosine similarity score

    Why cosine similarity and not Euclidean distance?
    - Our embeddings are L2-normalized (normalize_embeddings=True in indexer)
    - For normalized vectors, cosine similarity = dot product
    - pgvector's <=> operator computes cosine distance (1 - cosine similarity)
    - Cosine similarity is scale-invariant — direction matters, not magnitude
    """
    model = get_model()

    query_embedding = model.encode(query, normalize_embeddings=True)
    embedding_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding.tolist()) + "]"

    # Fix 3 — error handling on database failures
    try:
        async with AsyncSessionLocal() as session:
            # Fix 1 — CTE computes distance once instead of three times
            # Previous version repeated the embedding in WHERE, ORDER BY, SELECT
            # causing pgvector to compute the distance three times per row
            result = await session.execute(
                text(f"""
                    WITH distances AS (
                        SELECT
                            chunk_text,
                            source_document,
                            chunk_index,
                            1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                        FROM rag_chunks
                    )
                    SELECT chunk_text, source_document, chunk_index, similarity
                    FROM distances
                    WHERE similarity >= :min_similarity
                    ORDER BY similarity DESC
                    LIMIT :top_k
                """),
                {
                    "min_similarity": min_similarity,
                    "top_k": top_k,
                }
            )
            rows = result.fetchall()

    except Exception as e:
        # Fix 3 — return empty list instead of crashing the caller
        # Caller (LangGraph agent) can handle empty retrieval gracefully
        logger.error("retrieval_failed", query=query[:50], error=str(e))
        return []

    chunks = [
        {
            "chunk_text": row[0],
            "source_document": row[1],
            "chunk_index": row[2],
            "similarity": round(float(row[3]), 4),
        }
        for row in rows
    ]

    logger.info(
        "retrieval_complete",
        query=query[:50],
        top_k=top_k,
        results_found=len(chunks),
        top_similarity=chunks[0]["similarity"] if chunks else 0,
    )

    return chunks


async def test_retrieval() -> None:
    """Quick sanity check of the retrieval pipeline."""
    test_queries = [
        "adventure activities hiking mountains",
        "relaxing beach spa wellness",
        "budget backpacker cheap hostel street food",
        "luxury exclusive five star private villa",
        "cultural heritage temples museums history",
        "family friendly kids theme park aquarium",
    ]

    print("\n" + "="*60)
    print("RAG RETRIEVAL TEST")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-"*40)
        results = await retrieve(query, top_k=3)
        for r in results:
            dest = r["source_document"].replace("wikivoyage:", "")
            print(f"  {dest:<25} similarity={r['similarity']:.3f}")


if __name__ == "__main__":
    asyncio.run(test_retrieval())
