"""Async Redis client (singleton)."""

from __future__ import annotations

from redis.asyncio import Redis, from_url
from redis.asyncio.client import PubSub

from app.core.config import settings

_redis: Redis | None = None


async def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


async def publish(channel: str, message: str) -> None:
    """Publish a message to a Redis Pub/Sub channel."""
    r = await get_redis()
    await r.publish(channel, message)


async def get_pubsub() -> PubSub:
    """Return a new PubSub handle (caller is responsible for closing)."""
    r = await get_redis()
    return r.pubsub()


async def acquire_lock(key: str, timeout: int = 300) -> bool:
    """Acquire a distributed lock. Returns True if acquired, False if already locked."""
    r = await get_redis()
    # SETNX behavior: set only if it does not exist
    result = await r.set(key, "1", nx=True, ex=timeout)
    return bool(result)


async def release_lock(key: str) -> None:
    """Release the distributed lock."""
    r = await get_redis()
    await r.delete(key)
