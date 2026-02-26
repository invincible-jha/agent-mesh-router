"""AdaptiveThrottle — token-bucket rate limiter with backpressure awareness.

The throttle maintains a per-agent token bucket.  Each call to ``acquire``
consumes one token.  Tokens refill at a configurable rate (tokens per second).
When backpressure is high (pressure_score >= thresholds), the effective
refill rate is automatically reduced to slow down producers.

Backpressure adaptation
-----------------------
* Below warn threshold  → full rate.
* At/above warn threshold → rate scaled down proportionally between
  ``min_rate_fraction`` and 1.0.
* At/above critical threshold → rate capped at ``min_rate_fraction``
  (maximum throttling).

Formula::

    if pressure >= critical_threshold:
        effective_rate = base_rate * min_rate_fraction
    elif pressure >= warn_threshold:
        t = (pressure - warn_threshold) / (critical_threshold - warn_threshold)
        effective_rate = base_rate * (1 - t * (1 - min_rate_fraction))
    else:
        effective_rate = base_rate

Thread-safety
-------------
All state mutations use a ``threading.Lock`` so the throttle is safe to
call from multiple threads or concurrent asyncio tasks (with thread
executor).

Example
-------
::

    from agent_mesh_router.backpressure.throttle import AdaptiveThrottle
    from agent_mesh_router.backpressure.monitor import BackpressureMonitor

    monitor = BackpressureMonitor()
    monitor.register_queue("broker", max_depth=1000)

    throttle = AdaptiveThrottle(
        agent_id="agent-a",
        base_rate=100.0,      # 100 tokens/sec at zero pressure
        bucket_capacity=200,
        monitor=monitor,
    )

    # Attempt to acquire a token (non-blocking)
    if throttle.try_acquire():
        print("Send message")
    else:
        print("Rate limited")
"""
from __future__ import annotations

import asyncio
import logging
import time

from agent_mesh_router.backpressure.monitor import (
    CRITICAL_THRESHOLD,
    WARN_THRESHOLD,
    BackpressureMonitor,
)

logger = logging.getLogger(__name__)


class AdaptiveThrottle:
    """Token-bucket rate limiter that adapts to backpressure levels.

    Parameters
    ----------
    agent_id:
        Logical identifier for the agent being throttled (used in logs).
    base_rate:
        Baseline token refill rate in tokens per second at zero pressure.
    bucket_capacity:
        Maximum number of tokens the bucket can hold.  The bucket starts
        full.
    monitor:
        Optional ``BackpressureMonitor`` instance.  When provided, the
        effective refill rate is scaled down under pressure.  When None,
        no adaptive behaviour is applied.
    min_rate_fraction:
        Minimum refill rate as a fraction of ``base_rate`` when pressure
        is at or above the critical threshold.  Must be in (0.0, 1.0].
        Defaults to 0.1 (10% of base rate).
    warn_threshold:
        Pressure threshold above which throttling begins.
        Defaults to ``WARN_THRESHOLD`` (0.7).
    critical_threshold:
        Pressure threshold at which ``min_rate_fraction`` is applied.
        Defaults to ``CRITICAL_THRESHOLD`` (0.9).
    """

    def __init__(
        self,
        agent_id: str,
        *,
        base_rate: float = 100.0,
        bucket_capacity: int = 200,
        monitor: BackpressureMonitor | None = None,
        min_rate_fraction: float = 0.1,
        warn_threshold: float = WARN_THRESHOLD,
        critical_threshold: float = CRITICAL_THRESHOLD,
    ) -> None:
        if base_rate <= 0:
            raise ValueError(f"base_rate must be positive, got {base_rate}.")
        if bucket_capacity <= 0:
            raise ValueError(f"bucket_capacity must be positive, got {bucket_capacity}.")
        if not (0.0 < min_rate_fraction <= 1.0):
            raise ValueError(
                f"min_rate_fraction must be in (0.0, 1.0], got {min_rate_fraction}."
            )

        self._agent_id = agent_id
        self._base_rate = base_rate
        self._bucket_capacity = float(bucket_capacity)
        self._monitor = monitor
        self._min_rate_fraction = min_rate_fraction
        self._warn_threshold = warn_threshold
        self._critical_threshold = critical_threshold

        # Bucket state
        self._tokens: float = float(bucket_capacity)  # Start full
        self._last_refill: float = time.monotonic()

        import threading
        self._lock = threading.Lock()

        # Metrics
        self._total_acquired: int = 0
        self._total_rejected: int = 0

    # ------------------------------------------------------------------
    # Token acquisition
    # ------------------------------------------------------------------

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Attempt to consume ``tokens`` from the bucket (non-blocking).

        Refills the bucket based on elapsed time and the effective rate,
        then checks if enough tokens are available.

        Parameters
        ----------
        tokens:
            Number of tokens to consume.  Default is 1.0.

        Returns
        -------
        bool
            True if the tokens were consumed; False if the bucket does not
            have enough tokens (caller should back off or drop the message).
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += 1
                return True
            else:
                self._total_rejected += 1
                logger.debug(
                    "Throttle[%s]: rejected (tokens=%.2f < needed=%.2f).",
                    self._agent_id,
                    self._tokens,
                    tokens,
                )
                return False

    async def acquire_async(
        self,
        tokens: float = 1.0,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Async version of ``try_acquire`` with optional wait.

        Polls the bucket in short intervals (10 ms) until tokens are
        available or ``timeout`` is exceeded.

        Parameters
        ----------
        tokens:
            Number of tokens to consume.
        timeout:
            Maximum seconds to wait.  ``None`` means wait indefinitely.

        Returns
        -------
        bool
            True when tokens are consumed; False if timeout is reached.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            if self.try_acquire(tokens):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.01)

    # ------------------------------------------------------------------
    # Rate introspection
    # ------------------------------------------------------------------

    def effective_rate(self) -> float:
        """Return the current effective refill rate (tokens/sec).

        Accounts for backpressure scaling when a monitor is attached.

        Returns
        -------
        float
            Effective token refill rate.
        """
        if self._monitor is None:
            return self._base_rate

        pressure = self._monitor.pressure_score()
        return self._compute_rate(pressure)

    def available_tokens(self) -> float:
        """Return the number of tokens currently in the bucket.

        Triggers a refill calculation without consuming any tokens.

        Returns
        -------
        float
            Available token count in [0.0, bucket_capacity].
        """
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def agent_id(self) -> str:
        """Identifier of the throttled agent."""
        return self._agent_id

    @property
    def base_rate(self) -> float:
        """Configured baseline refill rate."""
        return self._base_rate

    @property
    def bucket_capacity(self) -> float:
        """Maximum token bucket capacity."""
        return self._bucket_capacity

    @property
    def total_acquired(self) -> int:
        """Total successful token acquisitions."""
        return self._total_acquired

    @property
    def total_rejected(self) -> int:
        """Total rejected acquisition attempts."""
        return self._total_rejected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Refill the bucket based on elapsed time and effective rate.

        Must be called with ``self._lock`` held.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        pressure = self._monitor.pressure_score() if self._monitor else 0.0
        rate = self._compute_rate(pressure)

        new_tokens = elapsed * rate
        self._tokens = min(self._bucket_capacity, self._tokens + new_tokens)

    def _compute_rate(self, pressure: float) -> float:
        """Compute the effective rate given the current pressure score.

        Parameters
        ----------
        pressure:
            Current pressure score in [0.0, 1.0].

        Returns
        -------
        float
            Effective token refill rate in tokens per second.
        """
        if pressure >= self._critical_threshold:
            return self._base_rate * self._min_rate_fraction

        if pressure >= self._warn_threshold:
            range_size = self._critical_threshold - self._warn_threshold
            t = (pressure - self._warn_threshold) / range_size
            fraction = 1.0 - t * (1.0 - self._min_rate_fraction)
            return self._base_rate * fraction

        return self._base_rate

    def __repr__(self) -> str:
        return (
            f"AdaptiveThrottle("
            f"agent={self._agent_id!r}, "
            f"rate={self.effective_rate():.1f}/s, "
            f"tokens={self._tokens:.1f}/{self._bucket_capacity:.0f}, "
            f"acquired={self._total_acquired}, "
            f"rejected={self._total_rejected}"
            f")"
        )
