import re
import time
import json
import random
import hashlib
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
REQUEST_DELAY_MIN = 4.0
REQUEST_DELAY_MAX = 7.0

HEADERS = {
    "User-Agent": "SmartTravelPlanner/1.0 (educational project; bahaamehyeldine@gmail.com)"
}


def _cache_path(destination: str) -> Path:
    safe_name = hashlib.md5(destination.lower().encode()).hexdigest()
    return CACHE_DIR / f"{CACHE_VERSION}_{safe_name}.json"


def _extract_section(wikitext: str, section_name: str) -> str:
    pattern = rf"==+\s*{re.escape(section_name)}\s*==+(.*?)(?===+[^=]|\Z)"
    match = re.search(pattern, wikitext, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.TimeoutException),
    reraise=True,
)
def _fetch_from_api(destination: str, params: dict) -> dict:
    with httpx.Client(timeout=15.0, headers=HEADERS) as client:
        response = client.get(WIKIVOYAGE_API, params=params)
        response.raise_for_status()
        return response.json()


def fetch_article(destination: str, force_refresh: bool = False) -> dict:
    cache_file = _cache_path(destination)

    if cache_file.exists() and not force_refresh:
        logger.debug("wikivoyage_cache_hit", destination=destination)
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

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
        data = _fetch_from_api(destination, params)
        pages = data.get("query", {}).get("pages", [])

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

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
        logger.debug("rate_limit_delay", seconds=round(delay, 2))
        time.sleep(delay)

        return result

    except httpx.HTTPStatusError as e:
        logger.error("wikivoyage_http_error", destination=destination, status_code=e.response.status_code)
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}

    except httpx.TimeoutException:
        logger.error("wikivoyage_timeout", destination=destination)
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}

    except Exception as e:
        logger.error("wikivoyage_unexpected_error", destination=destination, error=str(e))
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
        return {"title": destination, "understand": "", "do": "", "sleep": "", "full_text": ""}
