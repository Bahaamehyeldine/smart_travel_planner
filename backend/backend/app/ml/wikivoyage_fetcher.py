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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

logger = structlog.get_logger(__name__)

CACHE_VERSION = "v1"

CACHE_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "raw"
    / "wikivoyage_cache"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
REQUEST_DELAY = 1.5

HEADERS = {
    "User-Agent": "SmartTravelPlanner/1.0 (educational project; bahaamehyeldine@gmail.com)"
}


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
      without deleting them — useful for debugging
    """
    safe_name = hashlib.md5(destination.lower().encode()).hexdigest()
    return CACHE_DIR / f"{CACHE_VERSION}_{safe_name}.json"


# ─────────────────────────────────────────────
# Section extraction
# ─────────────────────────────────────────────

def _extract_section(wikitext: str, section_name: str) -> str:
    """
    Extract a named section from Wikivoyage wikitext format.

    Why regex instead of line-by-line parsing?
    - Wikivoyage is inconsistent: some use ==Do==, others == Do ==, others ===Do===
    - re.DOTALL makes . match newlines too — captures multi-line section content
    - re.IGNORECASE handles ==understand== vs ==Understand==
    - (?===+[^=]|\Z) stops at the next section heading OR end of document
    """
    pattern = rf"==+\s*{re.escape(section_name)}\s*==+(.*?)(?===+[^=]|\Z)"
    match = re.search(pattern, wikitext, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────
# API fetch with retry logic
# ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.TimeoutException),
    reraise=True,
)
def _fetch_from_api(destination: str, params: dict) -> dict:
    """
    Fetches from Wikivoyage API with retry logic.
    Separated from fetch_article so retries don't re-check the cache.

    Retries up to 3 times on timeout, waiting 2s, 4s, 8s between attempts.
    Only retries on TimeoutException — HTTP errors (403, 404) won't retry
    since retrying won't fix a permission or missing article error.
    """
    with httpx.Client(timeout=15.0, headers=HEADERS) as client:
        response = client.get(WIKIVOYAGE_API, params=params)
        response.raise_for_status()
        return response.json()


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
    - Used in Phase 3 to generate embeddings
    - Embedding models have token limits (~512 tokens ≈ ~2000 chars)
    - 5000 chars captures the most informative introductory content
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
        # _fetch_from_api handles retries on timeout internally
        data = _fetch_from_api(destination, params)

        # Navigate the MediaWiki API response structure
        # Response looks like: {"query": {"pages": [{"title": ..., "revisions": [...]}]}}
        pages = data.get("query", {}).get("pages", [])

        # Handle missing articles — not every destination has a Wikivoyage page
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
                "understand": _extract_section(wikitext, "Understand"),
                "do": _extract_section(wikitext, "Do"),
                "sleep": _extract_section(wikitext, "Sleep"),
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
        logger.error(
            "wikivoyage_http_error",
            destination=destination,
            status_code=e.response.status_code,
        )
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}

    except httpx.TimeoutException:
        logger.error("wikivoyage_timeout", destination=destination)
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}

    except Exception as e:
        logger.error("wikivoyage_unexpected_error", destination=destination, error=str(e))
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}