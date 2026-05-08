"""Generic async REST helper with retry/back-off.

Centralised so every tool gets the same timeout, header policy, and
SeaweedFS sidechain hook for raw payloads (design doc §7.2.2).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

import httpx

from app.core.config import settings
from app.storage.redis_client import get_redis

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_USER_AGENT = "aidd-agent/0.1 (+https://example.com)"

# ---------------------------------------------------------------------------
# Process-level TTL cache for idempotent GET requests.
# Avoids redundant external API calls when multiple graph nodes query the
# same URL within a short window (e.g. ChEMBL / UniProt in drugs / pathway).
# ---------------------------------------------------------------------------
def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    params_str = json.dumps(params, sort_keys=True, default=str) if params else ""
    raw = f"{url}:{params_str}"
    md5_hash = hashlib.md5(raw.encode()).hexdigest()  # nosec: not used for crypto
    return f"api_cache:{md5_hash}"


async def query_rest_api(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 3,
    expect_json: bool = True,
) -> Any:
    """Issue an HTTP request and return parsed JSON (or raw text).

    Retries with exponential back-off on 429 / 5xx / network errors.
    Raises the last exception on exhaustion.

    GET requests with ``expect_json=True`` are cached in Redis to avoid
    redundant external API calls and rate-limiting.
    """
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    if headers:
        merged_headers.update(headers)

    # --- Redis TTL cache (GET + JSON only) ---
    cache_hit_key: str | None = None
    if method.upper() == "GET" and expect_json:
        cache_hit_key = _cache_key(url, params)
        try:
            redis = await get_redis()
            cached_value = await redis.get(cache_hit_key)
            if cached_value:
                return json.loads(cached_value)
        except Exception as e:
            logger.warning("Redis cache read failed for %s: %s", url, e)

    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.request(
                    method, url, params=params, json=json_body, headers=merged_headers
                )
                # 4xx client errors (except 429 rate-limit) are not retryable.
                # Raise immediately so we don't waste retry budget on 404 / 403 etc.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    resp.raise_for_status()
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                result = resp.json() if expect_json else resp.text
                # Store successful GET responses in the Redis cache.
                if cache_hit_key is not None:
                    try:
                        redis = await get_redis()
                        await redis.set(
                            cache_hit_key,
                            json.dumps(result, ensure_ascii=False),
                            ex=settings.API_CACHE_TTL_SECONDS,
                        )
                    except Exception as e:
                        logger.warning("Redis cache write failed for %s: %s", url, e)
                return result
            except httpx.HTTPStatusError as exc:
                # Non-retryable 4xx: propagate immediately without logging a retry warning.
                if exc.response is not None and 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                last_exc = exc
                if attempt == max_retries - 1:
                    break
                backoff = 2**attempt
                logger.warning(
                    "REST %s %s failed (attempt %d/%d): %s — retrying in %ds",
                    method,
                    url,
                    attempt + 1,
                    max_retries,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    break
                backoff = 2**attempt
                logger.warning(
                    "REST %s %s failed (attempt %d/%d): %s — retrying in %ds",
                    method,
                    url,
                    attempt + 1,
                    max_retries,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

    assert last_exc is not None
    raise last_exc
