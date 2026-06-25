"""
feature_extractor.py

Extracts numeric features from Wikivoyage article text for travel style classification.

Feature design philosophy:
- Phase 2a: keyword/count features (interpretable, auditable, fast)
- Phase 2b: embedding features added later as separate experiment
- This separation allows clean comparison in results.csv

Improvements over v1:
- CLASS_THRESHOLDS config dict replaces hardcoded threshold logic
- Pydantic input validation at function boundary
- build_feature_matrix split into smaller testable units
- Incremental progress saving prevents data loss on failure
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator

import structlog

from app.ml.wikivoyage_fetcher import fetch_article

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Configuration — all tuneable values in one place
# ─────────────────────────────────────────────

# Threshold for binary keyword feature per class
# "how many keyword matches = strong signal for this class?"
# Adventure/Relaxation/Culture require 3+ (activity-count rubric)
# Budget/Luxury/Family require 1+ (keyword-presence rubric)
CLASS_THRESHOLDS = {
    "Adventure": 3,
    "Relaxation": 3,
    "Culture": 3,
    "Budget": 1,
    "Luxury": 1,
    "Family": 1,
}

# ─────────────────────────────────────────────
# Keyword dictionaries
# ─────────────────────────────────────────────

ADVENTURE_KEYWORDS = [
    "hiking", "trekking", "trek", "trails", "trail running", "backpacking routes",
    "diving", "scuba", "reef diving", "underwater", "freediving", "wreck diving",
    "climbing", "rock climbing", "mountaineering", "via ferrata", "bouldering", "ice climbing",
    "rafting", "kayaking", "white-water", "canoeing", "river rapids", "gorge swimming",
    "bungee", "cliff jumping", "canyon swinging",
    "skydiving", "paragliding", "parachuting", "base jumping", "hang gliding",
    "skiing", "snowboarding", "snow sports", "off-piste", "heli-skiing",
    "safari", "game drives", "wildlife spotting", "jungle trekking",
    "summit", "glacier", "alpine", "peak", "expedition",
]

RELAXATION_KEYWORDS = [
    "meditation", "mindfulness", "silent retreat", "vipassana",
    "spa", "spa treatments", "thermal spa", "ayurvedic", "hammam",
    "massage", "therapeutic massage", "thai massage", "hot stone",
    "yoga", "yoga retreat", "ashtanga", "hatha",
    "hot springs", "thermal baths", "onsen", "mineral baths", "geothermal",
    "wellness", "wellness retreat", "detox", "holistic", "healing",
    "sound healing", "sound bath", "gong bath",
    "beach lounging", "sunbathing", "secluded", "hammock", "calm beach",
    "sauna", "steam bath", "banya", "sento",
]

CULTURE_KEYWORDS = [
    "museum", "art gallery", "exhibition", "heritage museum",
    "ruins", "archaeological", "ancient", "old town", "castle", "fortress",
    "festival", "ceremony", "traditional", "folk festival", "religious festival",
    "market", "food market", "night market", "spice market",
    "artisan", "craft", "pottery", "weaving", "workshop",
    "cuisine", "cooking class", "food tour", "tasting tour",
    "temple", "shrine", "pagoda", "mosque", "cathedral", "heritage site", "unesco",
    "indigenous", "ethnic", "hill tribe", "homestay", "village visit",
    "architecture", "colonial", "medieval", "baroque", "renaissance",
]

BUDGET_KEYWORDS = [
    "affordable", "budget", "cheap", "inexpensive", "low-cost", "wallet-friendly",
    "backpacker", "hostel", "dorm", "guesthouse", "shoestring",
    "street food", "local food", "cheap eats", "value for money",
    "economical", "bargain", "frugal",
]

LUXURY_KEYWORDS = [
    "luxury", "exclusive", "five-star", "5-star", "upscale", "premium",
    "private villa", "overwater bungalow", "private pool", "butler service",
    "michelin", "fine dining", "haute cuisine",
    "opulent", "lavish", "indulgent", "bespoke", "curated",
    "high-end", "upmarket", "boutique resort", "designer hotel",
    "yacht", "helicopter", "concierge",
]

FAMILY_KEYWORDS = [
    "family-friendly", "kid-friendly", "kids", "children", "stroller",
    "theme park", "amusement park", "water park",
    "zoo", "aquarium", "wildlife park", "petting farm",
    "playground", "kids club", "family rooms", "all-ages",
    "interactive museum", "science museum", "hands-on",
    "shallow", "calm waters", "safe swimming", "family beach",
]

CLASS_KEYWORDS = {
    "Adventure": ADVENTURE_KEYWORDS,
    "Relaxation": RELAXATION_KEYWORDS,
    "Culture": CULTURE_KEYWORDS,
    "Budget": BUDGET_KEYWORDS,
    "Luxury": LUXURY_KEYWORDS,
    "Family": FAMILY_KEYWORDS,
}

REGIONS = [
    "Southeast Asia", "East Asia", "South Asia", "Central Asia",
    "Western Europe", "Eastern Europe", "Southern Europe", "Northern Europe",
    "North America", "Central America", "South America", "Caribbean",
    "North Africa", "West Africa", "East Africa", "Southern Africa",
    "Middle East", "Oceania",
]


# ─────────────────────────────────────────────
# Improvement 2 — Pydantic input validation
# ─────────────────────────────────────────────

class DestinationInput(BaseModel):
    """
    Validates inputs to extract_features at the function boundary.
    Pydantic is the fence — data is validated when it crosses in.

    Why validate here?
    - destination must be non-empty (empty string would return empty features)
    - region must be one of our known regions (unknown region = all zeros in one-hot)
    - Catching bad input early gives a clear error message vs silent wrong results
    """
    destination: str
    region: str
    force_refresh: bool = False

    @field_validator("destination")
    @classmethod
    def destination_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("destination cannot be empty")
        return v.strip()

    @field_validator("region")
    @classmethod
    def region_must_be_known(cls, v: str) -> str:
        if v not in REGIONS:
            raise ValueError(
                f"Unknown region '{v}'. Must be one of: {REGIONS}"
            )
        return v


# ─────────────────────────────────────────────
# Improvement 3 — Smaller, testable units
# ─────────────────────────────────────────────

def _count_keywords(text: str, keywords: list[str]) -> int:
    """Count keyword matches in text — case insensitive."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _extract_price_tier(sleep_text: str) -> int:
    """
    Extract price tier from Sleep section.
    Returns 1=Budget, 2=Mid-range, 3=Luxury.
    """
    if not sleep_text:
        return 2
    budget_score = _count_keywords(sleep_text, BUDGET_KEYWORDS)
    luxury_score = _count_keywords(sleep_text, LUXURY_KEYWORDS)
    if luxury_score > budget_score and luxury_score >= 2:
        return 3
    elif budget_score > luxury_score and budget_score >= 2:
        return 1
    return 2


def _compute_keyword_features(combined_text: str) -> dict:
    """
    Compute all keyword-based features from combined article text.
    Separated from extract_features so it can be tested in isolation.

    Returns dict of count, binary, and ratio features per class.
    """
    features = {}

    # Count and binary features per class
    for class_name, keywords in CLASS_KEYWORDS.items():
        count = _count_keywords(combined_text, keywords)
        key = class_name.lower()
        threshold = CLASS_THRESHOLDS[class_name]  # from config, not hardcoded
        features[f"{key}_keyword_count"] = count
        features[f"{key}_keyword_binary"] = int(count >= threshold)

    # Cross-class ratio features
    total = sum(
        features[f"{k.lower()}_keyword_count"]
        for k in CLASS_KEYWORDS
    ) or 1
    for class_name in CLASS_KEYWORDS:
        key = class_name.lower()
        features[f"{key}_keyword_ratio"] = (
            features[f"{key}_keyword_count"] / total
        )

    return features


def _compute_region_features(region: str) -> dict:
    """
    One-hot encode region as binary features.
    Separated so it can be tested in isolation.
    """
    return {
        f"region_{r.lower().replace(' ', '_')}": int(region == r)
        for r in REGIONS
    }


def extract_features(
    destination: str,
    region: str,
    force_refresh: bool = False,
) -> Optional[dict]:
    """
    Extract all features for a single destination.
    Validates inputs via Pydantic before any processing.
    """
    # Improvement 2 — validate at the boundary
    try:
        validated = DestinationInput(
            destination=destination,
            region=region,
            force_refresh=force_refresh,
        )
    except Exception as e:
        logger.error("invalid_input", destination=destination, region=region, error=str(e))
        return None

    article = fetch_article(validated.destination, force_refresh=validated.force_refresh)
    combined_text = f"{article['understand']} {article['do']}"

    if not combined_text.strip():
        logger.warning("empty_article_text", destination=validated.destination)

    # Improvement 3 — compose from smaller units
    features = {}
    features.update(_compute_keyword_features(combined_text))
    features["price_tier"] = _extract_price_tier(article["sleep"])
    features.update(_compute_region_features(validated.region))

    return features


# ─────────────────────────────────────────────
# Improvement 4 — Incremental saving
# ─────────────────────────────────────────────

def _load_progress(progress_path: Path) -> dict:
    """Load previously extracted features from progress file."""
    if progress_path.exists():
        with open(progress_path, "r") as f:
            return json.load(f)
    return {}


def _save_progress(progress_path: Path, progress: dict) -> None:
    """Save progress to disk after each successful extraction."""
    with open(progress_path, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def extract_single_row(row: pd.Series) -> Optional[dict]:
    """
    Extract features for one DataFrame row.
    Returns feature dict with label and name, or None on failure.
    Separated so it can be unit tested with a mock row.
    """
    destination = row["destination_name"]
    region = row["region"]
    label = row["travel_style_label"]

    features = extract_features(destination, region)
    if features is None:
        return None

    features["destination_name"] = destination
    features["label"] = label
    return features


def build_feature_matrix(
    csv_path: Path,
    progress_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Process all destinations and build feature matrix.

    Improvement 4: saves progress after each destination so a crash
    at destination 150 doesn't lose the first 149 results.

    Args:
        csv_path: path to destinations_labeled.csv
        progress_path: optional path to save/resume progress JSON
                       defaults to same directory as csv_path
    """
    df = pd.read_csv(csv_path)
    logger.info("building_feature_matrix", total_destinations=len(df))

    # Default progress file location
    if progress_path is None:
        progress_path = csv_path.parent / "feature_extraction_progress.json"

    # Load any previously completed extractions
    progress = _load_progress(progress_path)
    logger.info("resuming_from_progress", completed=len(progress))

    failed = []

    for idx, row in df.iterrows():
        destination = row["destination_name"]

        # Skip already completed destinations
        if destination in progress:
            logger.debug("skipping_cached", destination=destination)
            continue

        logger.info(
            "extracting_features",
            destination=destination,
            progress=f"{idx+1}/{len(df)}",
        )

        result = extract_single_row(row)

        if result is None:
            failed.append(destination)
            continue

        # Save progress after each success — Improvement 4
        progress[destination] = result
        _save_progress(progress_path, progress)

    if failed:
        logger.warning("feature_extraction_failed", count=len(failed), destinations=failed)

    # Build final DataFrame from all completed extractions
    rows = list(progress.values())
    feature_df = pd.DataFrame(rows)

    logger.info(
        "feature_matrix_complete",
        shape=str(feature_df.shape),
        failed_count=len(failed),
    )

    return feature_df