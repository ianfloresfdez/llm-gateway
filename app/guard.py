"""Skip a dependency that is down instead of paying its timeout on every request.

Redis and the embedding server are optional: cache and limits fail open. Without this, a gateway
started without Redis waited for the connect timeout (and logged a traceback) several times per
request. After one connection failure the dependency is skipped for `cooldown_s`, then retried.
"""

from __future__ import annotations

import logging
import time

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

log = logging.getLogger(__name__)

# Failures that mean "not reachable" (as opposed to a bad command or bad data)
UNREACHABLE = (RedisConnectionError, RedisTimeoutError, ConnectionError, TimeoutError, OSError)


class Guard:
    def __init__(self, name: str, cooldown_s: float = 30.0):
        self.name = name
        self.cooldown_s = cooldown_s
        self.down_until = 0.0

    @property
    def usable(self) -> bool:
        return time.monotonic() >= self.down_until

    def failed(self, exc: BaseException) -> None:
        """Record a failure. Logs once per outage, without a traceback."""
        if self.usable:
            log.warning(
                "%s unavailable (%s): skipping it for %.0f s", self.name, type(exc).__name__, self.cooldown_s
            )
        self.down_until = time.monotonic() + self.cooldown_s
