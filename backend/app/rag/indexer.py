"""
indexer.py

Indexes Wikivoyage destination articles into pgvector for RAG retrieval.

Pipeline:
1. Load cached Wikivoyage articles
2. Chunk each article into meaningful segments
3. Embed each chunk with all-MiniLM-L6-v2
4. Store chunks + embeddings in rag_chunks table

Run from backend/ directory:
    python -m app.rag.indexer
"""

import json
import hashlib
import asyncio
from pathlib import Path

import structlog
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from app.core.database import AsyncSessionLocal

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent.parent
CACHE_DIR = BASE_DIR / "data" / "raw" / "wikivoyage_cache"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "destinations_labeled.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _load_article(destination: str) -> dict:
    """Load cached Wikivoyage article for a destination."""
    safe_name = hashlib.md5(destination.lower().encode()).hexdigest()
    cache_file = CACHE_DIR / f"v1_{safe_name}.json"
    if not cache_file.exists():
        return {}
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _chunk_article(destination: str, article: dict) -> list[dict]:
    """
    Split article into chunks for indexing.

    Chunking strategy:
    - Each meaningful section becomes one chunk
    - We use Understand, Do, and Sleep sections separately
    - This gives the retriever section-level granularity
    - A query about "adventure activities" retrieves the Do section
      rather than a diluted full-article chunk

    Why not character-based chunking?
    - Our articles are already naturally segmented by Wikivoyage sections
    - Section-based chunks preserve semantic coherence
    - Simpler and more interpretable than sliding window approaches

    Fix 4: renamed loop variable from 'text' to 'section_text'
    to avoid shadowing the sqlalchemy text() import.
    """
    chunks = []

    sections = {
        "understand": article.get("understand", ""),
        "do": article.get("do", ""),
        "sleep": article.get("sleep", ""),
    }

    for idx, (section_name, section_text) in enumerate(sections.items()):
        # Fix 4: was 'text' — now 'section_text' to avoid shadowing sqlalchemy import
        if not section_text or len(section_text.strip()) < 50:
            continue

        # Prepend destination name and section for context
        # This helps the embedding capture "what place, what aspect"
        chunk_text = f"{destination} - {section_name.title()}: {section_text[:1500]}"

        chunks.append({
            "destination": destination,
            "section": section_name,
            "chunk_text": chunk_text,
            "chunk_index": idx,
            "source_document": f"wikivoyage:{destination}",
        })

    # If no sections found, use full_text as fallback
    if not chunks and article.get("full_text"):
        chunks.append({
            "destination": destination,
            "section": "full",
            "chunk_text": f"{destination}: {article['full_text'][:1500]}",
            "chunk_index": 0,
            "source_document": f"wikivoyage:{destination}",
        })

    return chunks


async def _clear_existing_chunks(session) -> None:
    """Clear existing rag_chunks before re-indexing."""
    await session.execute(text("DELETE FROM rag_chunks"))
    await session.commit()
    logger.info("cleared_existing_chunks")


async def _insert_chunk(
    session,
    chunk_text: str,
    embedding: list[float],
    source_document: str,
    chunk_index: int,
) -> None:
    """
    Insert a single chunk with its embedding into pgvector.

    Why format embedding directly into SQL string?
    asyncpg doesn't support ::vector cast with named parameters —
    it conflicts with SQLAlchemy's parameter substitution syntax.
    We safely format the embedding array directly into the SQL string.
    This is not a SQL injection risk because the embedding contains
    only floats produced by our own model — no user input involved.
    """
    embedding_str = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
    await session.execute(
        text(f"""
            INSERT INTO rag_chunks (chunk_text, embedding, source_document, chunk_index)
            VALUES (:chunk_text, '{embedding_str}'::vector, :source_document, :chunk_index)
        """),
        {
            "chunk_text": chunk_text,
            "source_document": source_document,
            "chunk_index": chunk_index,
        }
    )


async def build_index(force_rebuild: bool = False) -> None:
    """
    Build the RAG index from cached Wikivoyage articles.

    Args:
        force_rebuild: if True, clear existing chunks and rebuild from scratch
    """
    import pandas as pd

    df = pd.read_csv(FEATURES_PATH)
    destinations = df['destination_name'].tolist()
    logger.info("starting_indexing", n_destinations=len(destinations))

    # Load embedding model
    logger.info("loading_model", model=MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # Build all chunks first
    all_chunks = []
    skipped = []

    for dest in destinations:
        article = _load_article(dest)
        if not article:
            skipped.append(dest)
            continue
        chunks = _chunk_article(dest, article)
        all_chunks.extend(chunks)

    logger.info(
        "chunks_built",
        total_chunks=len(all_chunks),
        skipped=len(skipped),
    )

    if skipped:
        logger.warning("destinations_skipped", destinations=skipped)

    # Generate embeddings in batch
    logger.info("generating_embeddings", n_chunks=len(all_chunks))
    texts = [c["chunk_text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Store in pgvector
    logger.info("storing_in_pgvector", n_chunks=len(all_chunks))
    async with AsyncSessionLocal() as session:
        if force_rebuild:
            await _clear_existing_chunks(session)

        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            await _insert_chunk(
                session,
                chunk_text=chunk["chunk_text"],
                embedding=embedding.tolist(),
                source_document=chunk["source_document"],
                chunk_index=chunk["chunk_index"],
            )

            if (i + 1) % 50 == 0:
                await session.commit()
                logger.info("progress", inserted=i + 1, total=len(all_chunks))

        await session.commit()
        logger.info("indexing_complete", total_inserted=len(all_chunks))

    # Verify
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM rag_chunks"))
        count = result.scalar()
        logger.info("verification", total_chunks_in_db=count)
        print(f"\n✅ RAG index built: {count} chunks in pgvector")


if __name__ == "__main__":
    asyncio.run(build_index(force_rebuild=True))
