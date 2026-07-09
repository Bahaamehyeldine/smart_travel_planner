"""
graph.py

LangGraph agent for Smart Travel Planner.

Graph structure:
    user_input
        ↓
    [retrieve_node] — searches pgvector for relevant chunks
        ↓
    [classify_node] — predicts travel style from query keywords
        ↓
    [generate_node] — calls Groq LLM with retrieved context
        ↓
    response

Conditional edge:
    If retrieval returns no results → skip to generate with fallback message
"""

import asyncio
import joblib
from pathlib import Path
from typing import TypedDict, Annotated
from functools import lru_cache
import operator

import structlog
from langgraph.graph import StateGraph, END
from groq import Groq

from app.core.config import get_settings
from app.rag.retriever import retrieve

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"


# ─────────────────────────────────────────────
# Module-level singletons
# Fix 1 & 5: load model and Groq client once at startup
# not on every node invocation
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_classifier():
    """
    Load best saved model once — cached after first call.
    Fix 1: was called inside classify_node on every invocation.
    lru_cache ensures joblib.load runs exactly once.
    """
    candidates = [
        MODELS_DIR / "GradientBoosting_phase2b_v1.joblib",
        MODELS_DIR / "RandomForest_phase2b_v1.joblib",
        MODELS_DIR / "LogisticRegression_phase2a_v1.joblib",
        MODELS_DIR / "RandomForest_phase2a_v1.joblib",
    ]
    for path in candidates:
        if path.exists():
            logger.info("loaded_classifier", path=str(path))
            return joblib.load(path)
    logger.warning("no_classifier_found", searched=str(MODELS_DIR))
    return None


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """
    Instantiate Groq client once — cached after first call.
    Fix 5: was instantiated inside generate_node on every invocation.
    """
    settings = get_settings()
    logger.info("groq_client_initialized")
    return Groq(api_key=settings.GROQ_API_KEY)


# ─────────────────────────────────────────────
# Agent State
# Fix 4: all fields have defaults so callers only need to pass query
# ─────────────────────────────────────────────

class TravelPlannerState(TypedDict, total=False):
    """
    State that flows through the LangGraph graph.

    Each node receives the full state and returns a partial update.
    LangGraph merges updates using the reducer (operator.add for lists).

    Fix 4: total=False means all fields are optional with defaults.
    Callers only need to provide 'query' — all other fields
    are populated by nodes as the graph executes.

    Why TypedDict?
    - Type safety — each field has a declared type
    - LangGraph requires state to be a TypedDict or dataclass
    - Makes the data contract between nodes explicit
    """
    query: str
    retrieved_chunks: list[dict]
    predicted_style: str
    style_confidence: float
    messages: Annotated[list, operator.add]
    response: str
    error: str


# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

async def retrieve_node(state: TravelPlannerState) -> dict:
    """
    Node 1: Retrieve relevant chunks from pgvector.

    Takes the user query, embeds it, and retrieves the top-5
    most semantically similar chunks from our Wikivoyage index.
    """
    query = state["query"]
    logger.info("retrieve_node", query=query[:50])

    try:
        chunks = await retrieve(query, top_k=5, min_similarity=0.2)
        logger.info("retrieved_chunks", n=len(chunks))
        return {"retrieved_chunks": chunks}
    except Exception as e:
        logger.error("retrieve_node_error", error=str(e))
        return {"retrieved_chunks": [], "error": str(e)}


def classify_node(state: TravelPlannerState) -> dict:
    """
    Node 2: Predict travel style from query using ML classifier.

    Why we reconstruct features manually from query text:
    - extract_features() in feature_extractor.py expects a destination
      name and fetches its Wikivoyage article from cache
    - For a user query like "hiking in mountains", there is no cache entry
    - So we apply the same keyword-counting logic directly to the query string
    - This is intentional divergence from training — documented here explicitly

    Fix 3: feature column alignment uses model.feature_names_in_ when
    available, rather than assuming padding order by coincidence.
    """
    query = state["query"]
    logger.info("classify_node", query=query[:50])

    model = get_classifier()
    if model is None:
        return {"predicted_style": "Unknown", "style_confidence": 0.0}

    try:
        from app.ml.feature_extractor import (
            CLASS_KEYWORDS, CLASS_THRESHOLDS, REGIONS,
            _count_keywords,
        )
        import pandas as pd

        # Build keyword features from query text
        # Same logic as feature_extractor._compute_keyword_features()
        # but applied to raw query string instead of Wikivoyage article
        features = {}
        for class_name, keywords in CLASS_KEYWORDS.items():
            count = _count_keywords(query, keywords)
            key = class_name.lower()
            threshold = CLASS_THRESHOLDS[class_name]
            features[f"{key}_keyword_count"] = count
            features[f"{key}_keyword_binary"] = int(count >= threshold)

        total = sum(
            features[f"{k.lower()}_keyword_count"] for k in CLASS_KEYWORDS
        ) or 1
        for class_name in CLASS_KEYWORDS:
            key = class_name.lower()
            features[f"{key}_keyword_ratio"] = (
                features[f"{key}_keyword_count"] / total
            )

        features["price_tier"] = 2  # default mid-range — unknown for query

        # Region features — all zero for a query (no region specified)
        for r in REGIONS:
            features[f"region_{r.lower().replace(' ', '_')}"] = 0

        X = pd.DataFrame([features])

        # Fix 3 — align columns using model's actual feature names
        # This handles Phase 2b models that include emb_pca_* columns
        # Previous version assumed padding order by coincidence
        if hasattr(model, 'feature_names_in_'):
            expected_cols = list(model.feature_names_in_)
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[expected_cols]
        elif hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            while len(X.columns) < n_features:
                X[f"emb_pca_{len(X.columns) - 37}"] = 0.0

        proba = model.predict_proba(X)[0]
        classes = model.classes_
        predicted_idx = proba.argmax()
        predicted_style = classes[predicted_idx]
        confidence = float(proba[predicted_idx])

        logger.info("classification_result",
                    style=predicted_style,
                    confidence=round(confidence, 3))

        return {
            "predicted_style": predicted_style,
            "style_confidence": confidence,
        }

    except Exception as e:
        logger.error("classify_node_error", error=str(e))
        return {"predicted_style": "Unknown", "style_confidence": 0.0}


async def generate_node(state: TravelPlannerState) -> dict:
    """
    Node 3: Generate recommendation using Groq LLM.

    Assembles context from:
    - Retrieved RAG chunks (relevant destination descriptions)
    - Predicted travel style from ML classifier
    - Original user query

    Passes all of this to Llama 3.1 via Groq API and returns
    a structured travel recommendation.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    predicted_style = state.get("predicted_style", "Unknown")
    confidence = state.get("style_confidence", 0.0)

    logger.info("generate_node", query=query[:50],
                n_chunks=len(chunks), style=predicted_style)

    # Build context from retrieved chunks
    if chunks:
        context_parts = []
        for chunk in chunks[:4]:
            dest = chunk["source_document"].replace("wikivoyage:", "")
            similarity = chunk["similarity"]
            context_parts.append(
                f"[{dest} - similarity: {similarity:.2f}]\n"
                f"{chunk['chunk_text'][:400]}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "No specific destination information available."

    system_prompt = """You are an expert travel advisor for the Smart Travel Planner.
Your role is to recommend travel destinations based on the user's preferences.
Be specific, enthusiastic, and helpful. Keep recommendations concise but informative.
Always explain WHY a destination matches the user's travel style."""

    user_prompt = f"""User Query: {query}

Detected Travel Style: {predicted_style} (confidence: {confidence:.0%})

Relevant Destination Information:
{context}

Based on the user's query and the destination information above, provide:
1. Top 2-3 destination recommendations with brief explanations
2. Why these match their travel style
3. One practical tip for each destination

Keep the response friendly and under 300 words."""

    try:
        client = get_groq_client()

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        response = completion.choices[0].message.content
        logger.info("generation_complete", response_len=len(response))

        return {
            "response": response,
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        }

    except Exception as e:
        logger.error("generate_node_error", error=str(e))
        fallback = (
            f"I found some {predicted_style} destinations that might "
            f"interest you, but I'm having trouble generating a detailed "
            f"response right now. Please try again."
        )
        return {"response": fallback, "error": str(e)}


# ─────────────────────────────────────────────
# Conditional edge
# ─────────────────────────────────────────────

def should_generate(state: TravelPlannerState) -> str:
    """
    Conditional edge after retrieve_node.

    Routes to classify regardless of retrieval results.
    Even with no chunks, the LLM can give a general recommendation
    based on the predicted travel style alone.

    Extension point: route to 'clarify' node if both retrieval
    and classification return low confidence.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        logger.warning("no_chunks_retrieved_continuing_anyway")
    return "classify"


# ─────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────

def build_graph():
    """
    Assemble the LangGraph StateGraph.

    Node order: retrieve → classify → generate → END

    Why this order?
    - Retrieve first: get context before classification
    - Classify second: label the travel intent
    - Generate last: synthesize everything into a response
    """
    workflow = StateGraph(TravelPlannerState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        should_generate,
        {"classify": "classify"}
    )
    workflow.add_edge("classify", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ─────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────

async def test_agent():
    """Quick end-to-end test of the agent."""
    graph = build_graph()

    test_queries = [
        "I want to go hiking and do adventure sports in the mountains",
        "Looking for a relaxing beach vacation with spa treatments",
        "Budget travel in Southeast Asia with street food",
    ]

    for query in test_queries:
        print("\n" + "="*60)
        print(f"Query: {query}")
        print("="*60)

        # Fix 4: only pass query — all other fields default
        result = await graph.ainvoke({"query": query})

        print(f"Travel Style: {result.get('predicted_style', 'Unknown')} "
              f"({result.get('style_confidence', 0):.0%})")
        print(f"Chunks retrieved: {len(result.get('retrieved_chunks', []))}")
        print(f"\nResponse:\n{result.get('response', 'No response generated')}")


if __name__ == "__main__":
    asyncio.run(test_agent())
