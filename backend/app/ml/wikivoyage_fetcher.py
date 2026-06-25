"""
wikivoyage_fetcher.py

Fetches and caches Wikivoyage articles for destination feature extraction.

Design decisions:
- Disk cache avoids redundant HTTP requests and ensures reproducibility
- Rate limiting respects Wikivoyage as a free community resource
- Cache versioning allows invalidation when extraction logic changes
- structlog replaces print statements per project engineering standards
"""

import re
import time
import json
import hashlib
from pathlib import Path

import httpx
import structlog

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# structlog logger — named after this module for easy filtering in logs
logger = structlog.get_logger(__name__)

# Cache version — bump this string whenever you change extraction logic
# This automatically invalidates all old cached files without deleting them
CACHE_VERSION = "v1"

# Path to cache directory — 4 .parent calls climb from this file up to project root
# this_file → ml/ → app/ → backend/ → project_root/
CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "raw"
    / "wikivoyage_cache"
)

# Create cache directory if it doesn't exist
# parents=True creates intermediate directories too
# exist_ok=True doesn't raise an error if it already exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Wikivoyage MediaWiki API endpoint
# We use the API rather than scraping HTML — more stable, structured response
WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"

# Delay between HTTP requests in seconds
# 1.5 seconds is respectful for a free community resource
# Too fast = risk of being blocked; too slow = dataset takes forever to build
REQUEST_DELAY = 1.5


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────

def _cache_path(destination: str) -> Path:
    """
    Generate a stable, filesystem-safe cache filename for a destination.

    Why MD5 hash instead of the destination name directly?
    - Destination names can contain characters invalid in filenames: '/', ':', "'"
      e.g. "Xi'an", "Turks and Caicos" — these break file paths on some OSes
    - MD5 produces a fixed-length hex string safe for any filesystem
    - .lower() ensures "Rome" and "rome" map to the same cache file
    - Cache version prefix means bumping CACHE_VERSION bypasses all old files
      without deleting them — useful for debugging (old files still there for comparison)
    """
    safe_name = hashlib.md5(destination.lower().encode()).hexdigest()
    return CACHE_DIR / f"{CACHE_VERSION}_{safe_name}.json"


# ─────────────────────────────────────────────
# Section extraction
# ─────────────────────────────────────────────

def _extract_section(wikitext: str, section_name: str) -> str:
    """
    Extract a named section from Wikivoyage wikitext format.

    Wikivoyage articles use MediaWiki markup where sections look like:
        == Understand ==
        text here...
        == Do ==
        more text...

    Why regex instead of line-by-line parsing?
    - Wikivoyage is inconsistent: some use ==Do==, others == Do ==, others ===Do===
    - re.DOTALL makes . match newlines too — captures multi-line section content
    - re.IGNORECASE handles ==understand== vs ==Understand==
    - (?===+[^=]|\Z) stops at the next section heading OR end of document
      [^=] prevents matching === (subsection) as a section boundary
    """
    pattern = rf"==+\s*{re.escape(section_name)}\s*==+(.*?)(?===+[^=]|\Z)"
    match = re.search(pattern, wikitext, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────
# Main fetch function
# ─────────────────────────────────────────────

def fetch_article(destination: str, force_refresh: bool = False) -> dict:
    """
    Fetch a Wikivoyage article for a destination.

    Returns a dict with extracted sections:
        {
            "title": str,
            "understand": str,   # character/vibe description
            "do": str,           # activities list
            "sleep": str,        # accommodation + price signals
            "full_text": str,    # first 5000 chars for Phase 3 embeddings
        }

    Args:
        destination: destination name matching Wikivoyage article title
        force_refresh: if True, bypass cache and re-fetch from API

    Why full_text is capped at 5000 characters:
    - This field will be used in Phase 3 to generate embeddings
    - Embedding models have token limits (~512 tokens ≈ ~2000 chars)
    - 5000 chars gives us the most informative part of each article
      (introductory sections) without storing entire articles in cache
    - Storing less = smaller cache files = faster cache reads
    """
    cache_file = _cache_path(destination)

    # ── Cache hit ──────────────────────────────
    # Return cached version if available and not forcing refresh
    # This is the hot path during development — most calls return immediately
    if cache_file.exists() and not force_refresh:
        logger.debug("wikivoyage_cache_hit", destination=destination)
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Cache miss — fetch from API ────────────
    logger.info("wikivoyage_fetching", destination=destination)

    # MediaWiki API parameters
    # action=query: we're reading, not writing
    # prop=revisions: get article content via revision history
    # rvprop=content: include the actual wikitext content
    # rvslots=main: get the main content slot (not talk page etc.)
    # formatversion=2: use modern API response format (cleaner JSON structure)
    params = {
        "action": "query",
        "titles": destination,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2",
    }

    try:
        # httpx.Client is the sync HTTP client
        # timeout=10.0 means: give up if no response within 10 seconds
        # We use sync here (not async) because this script runs as a
        # standalone data pipeline, not inside a FastAPI request handler
        with httpx.Client(timeout=10.0) as client:
            response = client.get(WIKIVOYAGE_API, params=params)

            # raise_for_status() raises an exception for 4xx/5xx HTTP errors
            # e.g. 404 Not Found, 429 Too Many Requests, 500 Server Error
            response.raise_for_status()
            data = response.json()

        # Navigate the MediaWiki API response structure
        # Response looks like: {"query": {"pages": [{"title": ..., "revisions": [...]}]}}
        pages = data.get("query", {}).get("pages", [])

        # Handle missing articles — not every destination has a Wikivoyage page
        # pages[0].get("missing") is True when the article doesn't exist
        if not pages or pages[0].get("missing"):
            logger.warning("wikivoyage_article_missing", destination=destination)
            result = {
                "title": destination,
                "understand": "",
                "do": "",
                "sleep": "",
                "full_text": "",
            }
        else:
            # Extract wikitext from the nested API response structure
            wikitext = pages[0]["revisions"][0]["slots"]["main"]["content"]

            result = {
                "title": destination,
                # "Understand" section = character/vibe description
                # Primary signal for our labeling rubric
                "understand": _extract_section(wikitext, "Understand"),
                # "Do" section = activities list
                # Used for activity-count features
                "do": _extract_section(wikitext, "Do"),
                # "Sleep" section = accommodation types and price ranges
                # Used for Budget/Luxury price-tier features
                "sleep": _extract_section(wikitext, "Sleep"),
                # First 5000 chars of full article for Phase 3 RAG embeddings
                # Captures the most informative introductory content
                "full_text": wikitext[:5000],
            }

            logger.info(
                "wikivoyage_fetch_success",
                destination=destination,
                understand_len=len(result["understand"]),
                do_len=len(result["do"]),
            )

        # ── Write to cache ─────────────────────
        # ensure_ascii=False preserves non-ASCII characters (e.g. "Fès", "Zürich")
        # indent=2 makes cache files human-readable for debugging
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # ── Rate limiting ──────────────────────
        # Sleep AFTER caching so a KeyboardInterrupt during sleep
        # doesn't lose the fetched data
        time.sleep(REQUEST_DELAY)

        return result

    except httpx.HTTPStatusError as e:
        # HTTP error (4xx, 5xx) — log with status code for debugging
        logger.error(
            "wikivoyage_http_error",
            destination=destination,
            status_code=e.response.status_code,
        )
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}

    except httpx.TimeoutException:
        # Request timed out — Wikivoyage sometimes slow
        logger.error("wikivoyage_timeout", destination=destination)
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}

    except Exception as e:
        # Catch-all for unexpected errors — log but don't crash the pipeline
        logger.error("wikivoyage_unexpected_error", destination=destination, error=str(e))
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}