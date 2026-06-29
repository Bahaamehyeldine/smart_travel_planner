"""
embedding_extractor.py

Generates sentence embeddings from Wikivoyage article text.
Used in Phase 2b to augment keyword features with semantic signal.

Model: all-MiniLM-L6-v2 (384 dimensions)
- Fast, lightweight, good for short-to-medium text
- Already installed via sentence-transformers

Why embeddings help:
- Keyword features miss characterization language
  e.g. "serene beaches perfect for unwinding" contains none of our
  relaxation keywords but semantically signals Relaxation strongly
- Embeddings capture semantic meaning regardless of exact wording

Why PCA is safe to fit on all data (not just training):
- PCA is an unsupervised transformation — it only looks at the
  distribution of embedding vectors, never at class labels
- This is fundamentally different from fitting a scaler on test data,
  which would shift/scale features based on test distribution
- Since embeddings are label-independent, fitting PCA on all 200
  samples does not leak any label information into the transformation
- We still fit PCA on training indices only (fix 1) to be strictly
  correct and consistent with sklearn best practices
"""

import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

import structlog
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent.parent
CACHE_DIR = BASE_DIR / "data" / "raw" / "wikivoyage_cache"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

EMBEDDING_VERSION = "v1"
EMBEDDINGS_PATH = PROCESSED_DIR / f"embeddings_{EMBEDDING_VERSION}.npy"
EMBEDDINGS_INDEX_PATH = PROCESSED_DIR / f"embeddings_{EMBEDDING_VERSION}_index.json"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Fix 3 — configurable constant instead of hardcoded default
PCA_N_COMPONENTS = 50


def _get_article_text(destination: str) -> str:
    """
    Load cached article text for a destination.
    Uses understand section primarily — most characterization language lives here.
    Falls back to full_text if understand is empty.
    Returns empty string if cache miss — caller handles fallback.
    """
    safe_name = hashlib.md5(destination.lower().encode()).hexdigest()
    cache_file = CACHE_DIR / f"v1_{safe_name}.json"

    if not cache_file.exists():
        logger.warning("cache_miss_for_embedding", destination=destination)
        return ""

    with open(cache_file, "r", encoding="utf-8") as f:
        article = json.load(f)

    understand = article.get("understand", "")
    full_text = article.get("full_text", "")

    text = understand if understand.strip() else full_text
    return text[:2000]  # cap at ~500 tokens for model limit


def build_embedding_matrix(
    destinations: list[str],
    force_refresh: bool = False,
) -> np.ndarray:
    """
    Build embedding matrix for all destinations.

    Returns:
        np.ndarray of shape (n_destinations, EMBEDDING_DIM)
        Row order matches destinations list.

    Caches to disk — subsequent calls load from cache unless force_refresh=True.
    """
    if EMBEDDINGS_PATH.exists() and EMBEDDINGS_INDEX_PATH.exists() and not force_refresh:
        logger.info("loading_cached_embeddings", path=str(EMBEDDINGS_PATH))
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(EMBEDDINGS_INDEX_PATH, "r") as f:
            index = json.load(f)
        try:
            ordered = np.array([embeddings[index[d]] for d in destinations])
            logger.info("embeddings_loaded", shape=str(ordered.shape))
            return ordered
        except KeyError as e:
            logger.warning("cache_index_mismatch_regenerating", missing=str(e))

    logger.info("loading_embedding_model", model=MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = []
    for dest in destinations:
        text = _get_article_text(dest)
        if not text:
            # Fix 2 — zero vector fallback instead of misleading placeholder text
            # A zero vector is honest: we have no information for this destination
            # A placeholder string like "travel destination X" would produce a
            # near-random embedding that adds noise to the feature matrix
            logger.warning("no_text_using_zero_vector", destination=dest)
            texts.append("")
        else:
            texts.append(text)

    # Encode non-empty texts, replace empties with zero vectors
    logger.info("generating_embeddings", n_destinations=len(texts))
    embeddings = np.zeros((len(texts), EMBEDDING_DIM))
    non_empty_indices = [i for i, t in enumerate(texts) if t]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    if non_empty_texts:
        encoded = model.encode(
            non_empty_texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        for i, idx in enumerate(non_empty_indices):
            embeddings[idx] = encoded[i]

    np.save(EMBEDDINGS_PATH, embeddings)
    index = {dest: i for i, dest in enumerate(destinations)}
    with open(EMBEDDINGS_INDEX_PATH, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    logger.info("embeddings_saved", path=str(EMBEDDINGS_PATH), shape=str(embeddings.shape))
    return embeddings


def add_embedding_features(
    df_features: pd.DataFrame,
    destinations: list[str],
    train_indices: np.ndarray,
    n_components: int = PCA_N_COMPONENTS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Add PCA-reduced embedding features to existing feature matrix.

    Fix 1 — PCA fitted on training indices only:
    We pass train_indices so PCA is fit exclusively on training rows.
    It is then applied (transformed) to all rows including test rows.
    This is the correct sklearn pattern: fit on train, transform on all.

    Why PCA:
    - 384 embedding dims > 200 samples → curse of dimensionality
    - PCA reduces to n_components capturing most variance
    - 50 components typically captures 80-90% variance for MiniLM
    - Result: 37 keyword features + 50 PCA = 87 total features

    Args:
        df_features: existing keyword feature matrix
        destinations: destination names in same row order as df_features
        train_indices: integer indices of training rows — PCA fit here only
        n_components: number of PCA components (default: PCA_N_COMPONENTS)
        force_refresh: regenerate embeddings even if cached
    """
    embeddings = build_embedding_matrix(destinations, force_refresh=force_refresh)

    # Fix 1 — fit scaler and PCA on training rows only
    train_embeddings = embeddings[train_indices]

    logger.info("fitting_pca_on_train", n_components=n_components,
                train_size=len(train_indices))

    scaler = StandardScaler()
    scaler.fit(train_embeddings)
    embeddings_scaled = scaler.transform(embeddings)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings_scaled[train_indices])
    embeddings_reduced = pca.transform(embeddings_scaled)

    explained_variance = pca.explained_variance_ratio_.sum()
    logger.info(
        "pca_complete",
        n_components=n_components,
        explained_variance=f"{explained_variance:.1%}",
    )

    emb_cols = [f"emb_pca_{i}" for i in range(n_components)]
    emb_df = pd.DataFrame(
        embeddings_reduced,
        columns=emb_cols,
        index=df_features.index
    )

    augmented = pd.concat([df_features, emb_df], axis=1)
    logger.info(
        "features_augmented",
        original=len(df_features.columns),
        added=n_components,
        total=len(augmented.columns),
    )

    return augmented
