"""Production Rate Limiter and Throttling Utilities for ImageJev.

Provides:
- In-memory sliding-window RateLimiter with zero external dependencies.
- ConcurrencyGate for bounding simultaneous CPU/GPU inference tasks.
- Client identification supporting reverse proxies (X-Forwarded-For, CF-Connecting-IP).
- Standard rate limit headers (X-RateLimit-Limit, Remaining, Reset, Retry-After).
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple


def get_client_ip(headers: Dict[str, str], client_host: Optional[str] = None) -> str:
    """Extract real client IP, respecting proxy headers when behind Cloudflare/Render/Nginx."""
    # Cloudflare Connecting IP
    cf_ip = headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()

    # Standard X-Forwarded-For (client, proxy1, proxy2...)
    x_forwarded = headers.get("x-forwarded-for")
    if x_forwarded:
        parts = x_forwarded.split(",")
        if parts and parts[0].strip():
            return parts[0].strip()

    # X-Real-IP
    real_ip = headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    return client_host or "127.0.0.1"


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding window rate limiter.

    Parameters:
        requests_per_minute: Default limit for window of 60 seconds.
        window_seconds: Duration of the sliding window in seconds (default: 60).
        enabled: Toggle to disable rate limiting (e.g. for unit tests or dev).
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        window_seconds: float = 60.0,
        enabled: bool = True,
    ):
        self.default_limit = requests_per_minute
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._history: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def check(
        self,
        client_key: str,
        limit: Optional[int] = None,
        cost: int = 1,
    ) -> Tuple[bool, int, int, int]:
        """Check if request is allowed under the sliding window.

        Args:
            client_key: Unique client identifier (e.g. IP or API key).
            limit: Specific limit for this route (falls back to default_limit).
            cost: Number of tokens consumed by this request (e.g. batch size).

        Returns:
            Tuple of:
                allowed (bool): True if allowed, False if rate limited.
                remaining (int): Remaining request capacity in window.
                retry_after (int): Seconds client must wait before retrying.
                reset_in (int): Seconds until current window fully resets.
        """
        if not self.enabled:
            return True, 999999, 0, 0

        max_allowed = limit if limit is not None else self.default_limit
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # Periodic cleanup of inactive client entries every 300 seconds
            if now - self._last_cleanup > 300.0:
                self._cleanup_stale(cutoff)
                self._last_cleanup = now

            timestamps = self._history[client_key]

            # Prune events older than the window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            current_usage = len(timestamps)

            if current_usage + cost <= max_allowed:
                # Record the new request(s)
                for _ in range(cost):
                    timestamps.append(now)
                remaining = max(0, max_allowed - (current_usage + cost))
                reset_in = int(self.window_seconds - (now - timestamps[0])) if timestamps else int(self.window_seconds)
                return True, remaining, 0, max(1, reset_in)
            else:
                # Rate limit exceeded
                oldest = timestamps[0] if timestamps else now
                retry_after = max(1, int(self.window_seconds - (now - oldest)))
                return False, 0, retry_after, retry_after

    def _cleanup_stale(self, cutoff: float) -> None:
        """Remove empty or expired deques to keep memory bounded."""
        stale_keys = [k for k, q in self._history.items() if not q or q[-1] <= cutoff]
        for k in stale_keys:
            del self._history[k]

    def reset(self) -> None:
        """Clear all rate limit state (useful in test teardown)."""
        with self._lock:
            self._history.clear()


class ConcurrencyGate:
    """Thread-safe concurrency throttle to prevent CPU/GPU compute exhaustion.

    Limits simultaneous heavy inferences to `max_concurrent`.
    Extra requests wait up to `timeout` seconds before failing with 503.
    """

    def __init__(self, max_concurrent: int = 4, timeout: float = 30.0):
        self.max_concurrent = max(1, max_concurrent)
        self.timeout = timeout
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)
        self._in_flight = 0
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        to = timeout if timeout is not None else self.timeout
        acquired = self._semaphore.acquire(blocking=True, timeout=to)
        if acquired:
            with self._lock:
                self._in_flight += 1
        return acquired

    def release(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
        self._semaphore.release()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight
