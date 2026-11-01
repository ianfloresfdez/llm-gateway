"""Per-key limits stored in Redis, so they hold across every gateway replica.

- requests per minute: fixed one-minute window (INCR is atomic, EXPIRE cleans up).
- tokens per day: counter incremented after each response, checked before the next.
Fails open if Redis is unreachable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from redis.asyncio import Redis

from app.config import KeyLimits
from app.guard import UNREACHABLE, Guard

log = logging.getLogger(__name__)


@dataclass
class LimitDecision:
    allowed: bool
    reason: str = ""
    retry_after_s: int = 0


class RateLimiter:
    def __init__(self, redis: Redis, guard: Guard | None = None):
        self.redis = redis
        self.guard = guard or Guard("redis")

    async def check(self, key_name: str, limits: KeyLimits) -> LimitDecision:
        now = time.time()
        minute = int(now // 60)
        day = time.strftime("%Y%m%d", time.gmtime(now))
        rpm_key = f"llmgw:rl:{key_name}:{minute}"
        if not self.guard.usable:
            return LimitDecision(True)
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.incr(rpm_key)
                pipe.expire(rpm_key, 120)
                pipe.get(f"llmgw:tok:{key_name}:{day}")
                count, _, used = await pipe.execute()
        except UNREACHABLE as e:
            self.guard.failed(e)
            return LimitDecision(True)
        except Exception:
            log.warning("rate limiter unavailable, failing open", exc_info=True)
            return LimitDecision(True)
        if count > limits.rpm:
            return LimitDecision(False, "rpm", retry_after_s=60 - int(now % 60))
        if used is not None and int(used) >= limits.tokens_per_day:
            seconds_to_midnight = 86400 - int(now % 86400)
            return LimitDecision(False, "tokens_per_day", retry_after_s=seconds_to_midnight)
        return LimitDecision(True)

    async def record_tokens(self, key_name: str, tokens: int) -> None:
        if tokens <= 0 or not self.guard.usable:
            return
        day = time.strftime("%Y%m%d", time.gmtime())
        key = f"llmgw:tok:{key_name}:{day}"
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.incrby(key, tokens)
                pipe.expire(key, 2 * 86400)
                await pipe.execute()
        except UNREACHABLE as e:
            self.guard.failed(e)
        except Exception:
            log.warning("could not record token usage", exc_info=True)
