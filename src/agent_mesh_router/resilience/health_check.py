"""HealthChecker — run health probes against service endpoints.

The health checker executes a user-supplied probe callable for each
``ServiceEndpoint``, records the result, and maintains a rolling history
per endpoint.  It does not start a background loop; callers are responsible
for scheduling checks (e.g. from a ``HealthChecker`` scheduler or the CLI).

Example
-------
::

    import httpx
    from agent_mesh_router.resilience.health_check import (
        HealthChecker,
        HealthCheckConfig,
        HealthCheckResult,
        HealthStatus,
    )
    from agent_mesh_router.resilience.load_balancer import ServiceEndpoint

    config = HealthCheckConfig(
        interval_seconds=30.0,
        timeout_seconds=5.0,
        healthy_threshold=2,
        unhealthy_threshold=3,
    )
    checker = HealthChecker(config)

    def probe(endpoint: ServiceEndpoint) -> None:
        httpx.get(f"http://{endpoint.host}:{endpoint.port}/health", timeout=5)

    endpoint = ServiceEndpoint("api", host="10.0.0.1", port=8080)
    result = checker.check(endpoint, probe)
    print(result.status)   # HealthStatus.HEALTHY or UNHEALTHY
    print(result.latency_ms)
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from agent_mesh_router.resilience.load_balancer import ServiceEndpoint

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_SIZE = 100


class HealthStatus(str, Enum):
    """Possible health statuses returned by a health check probe."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HealthCheckConfig:
    """Immutable configuration for a ``HealthChecker``.

    Attributes
    ----------
    interval_seconds:
        How often the caller should schedule checks (informational; the
        checker itself does not schedule anything).  Must be > 0.
    timeout_seconds:
        Maximum seconds allowed for a probe callable to complete before
        it is treated as unhealthy.  Must be > 0.
    healthy_threshold:
        Consecutive successful probes required to transition from
        UNHEALTHY to HEALTHY.  Must be >= 1.
    unhealthy_threshold:
        Consecutive failed probes required to transition from
        HEALTHY to UNHEALTHY.  Must be >= 1.
    max_history:
        Maximum number of ``HealthCheckResult`` records stored per
        endpoint.  Oldest records are dropped.  Defaults to 100.
    """

    interval_seconds: float
    timeout_seconds: float
    healthy_threshold: int
    unhealthy_threshold: int
    max_history: int = field(default=_DEFAULT_HISTORY_SIZE)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be > 0, got {self.interval_seconds}."
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}."
            )
        if self.healthy_threshold < 1:
            raise ValueError(
                f"healthy_threshold must be >= 1, got {self.healthy_threshold}."
            )
        if self.unhealthy_threshold < 1:
            raise ValueError(
                f"unhealthy_threshold must be >= 1, got {self.unhealthy_threshold}."
            )
        if self.max_history < 1:
            raise ValueError(
                f"max_history must be >= 1, got {self.max_history}."
            )


@dataclass(frozen=True)
class HealthCheckResult:
    """Result of a single health check probe.

    Attributes
    ----------
    endpoint_name:
        Name of the ``ServiceEndpoint`` that was probed.
    status:
        Health status determined by this probe.
    latency_ms:
        Round-trip time of the probe in milliseconds.
    checked_at:
        UTC datetime when the probe completed.
    details:
        Arbitrary key/value metadata (e.g. HTTP status code, error message).
    """

    endpoint_name: str
    status: HealthStatus
    latency_ms: float
    checked_at: datetime
    details: dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Execute health check probes and maintain per-endpoint history.

    Parameters
    ----------
    config:
        Immutable configuration object.
    """

    def __init__(self, config: HealthCheckConfig) -> None:
        self._config = config
        # endpoint_name → deque of HealthCheckResult
        self._history: dict[str, deque[HealthCheckResult]] = {}
        # Consecutive success/failure counters for threshold evaluation
        self._consecutive_successes: dict[str, int] = {}
        self._consecutive_failures: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        endpoint: ServiceEndpoint,
        probe: Callable[[ServiceEndpoint], None],
    ) -> HealthCheckResult:
        """Run a single health probe against ``endpoint``.

        The ``probe`` callable is expected to raise any exception to signal
        failure (e.g. ``ConnectionRefusedError``, ``TimeoutError``).  A
        clean return is interpreted as healthy.

        The probe is timed; if it exceeds ``config.timeout_seconds`` it is
        treated as UNHEALTHY.  (Actual async timeout enforcement requires
        the caller to use an async probe with ``asyncio.wait_for``; this
        synchronous wrapper uses a wall-clock heuristic only.)

        Parameters
        ----------
        endpoint:
            The endpoint to probe.
        probe:
            Callable that takes a ``ServiceEndpoint`` and returns nothing.
            Raises on failure.

        Returns
        -------
        HealthCheckResult
            Result of the probe.
        """
        start = time.monotonic()
        error_details: dict[str, Any] = {}
        probe_failed = False

        try:
            probe(endpoint)
        except Exception as exc:
            probe_failed = True
            error_details = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            logger.debug(
                "HealthChecker: probe failed for %r: %s: %s.",
                endpoint.name,
                type(exc).__name__,
                exc,
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Timeout heuristic: if the probe succeeded but took too long,
        # treat it as failed rather than healthy.
        if not probe_failed and elapsed_ms > self._config.timeout_seconds * 1000.0:
            probe_failed = True
            error_details = {
                "error_type": "TimeoutError",
                "error_message": (
                    f"Probe exceeded timeout of "
                    f"{self._config.timeout_seconds * 1000:.0f} ms "
                    f"(took {elapsed_ms:.1f} ms)."
                ),
            }

        status = self._compute_status(endpoint.name, not probe_failed)

        result = HealthCheckResult(
            endpoint_name=endpoint.name,
            status=status,
            latency_ms=elapsed_ms,
            checked_at=datetime.now(tz=timezone.utc),
            details=error_details,
        )

        self._record(endpoint.name, result)
        return result

    def check_all(
        self,
        endpoints: list[ServiceEndpoint],
        probe: Callable[[ServiceEndpoint], None],
    ) -> list[HealthCheckResult]:
        """Run a health probe against each endpoint in ``endpoints``.

        Parameters
        ----------
        endpoints:
            List of endpoints to probe sequentially.
        probe:
            Probe callable applied to each endpoint.

        Returns
        -------
        list[HealthCheckResult]
            Results in the same order as ``endpoints``.
        """
        return [self.check(ep, probe) for ep in endpoints]

    def history(self, endpoint_name: str) -> list[HealthCheckResult]:
        """Return stored check results for a named endpoint.

        Parameters
        ----------
        endpoint_name:
            The endpoint whose history to retrieve.

        Returns
        -------
        list[HealthCheckResult]
            Results in chronological order (oldest first).
            Returns an empty list if the endpoint has no recorded history.
        """
        return list(self._history.get(endpoint_name, deque()))

    def clear_history(self, endpoint_name: str | None = None) -> None:
        """Clear stored history.

        Parameters
        ----------
        endpoint_name:
            If provided, clear only this endpoint's history.  If None,
            clear history for all endpoints.
        """
        if endpoint_name is not None:
            self._history.pop(endpoint_name, None)
            self._consecutive_successes.pop(endpoint_name, None)
            self._consecutive_failures.pop(endpoint_name, None)
        else:
            self._history.clear()
            self._consecutive_successes.clear()
            self._consecutive_failures.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_status(self, name: str, success: bool) -> HealthStatus:
        """Update counters and return the resolved ``HealthStatus``.

        Applies threshold logic: consecutive failures must reach
        ``unhealthy_threshold`` before status becomes UNHEALTHY, and
        consecutive successes must reach ``healthy_threshold`` before
        status returns to HEALTHY.
        """
        if success:
            self._consecutive_successes[name] = (
                self._consecutive_successes.get(name, 0) + 1
            )
            self._consecutive_failures[name] = 0
        else:
            self._consecutive_failures[name] = (
                self._consecutive_failures.get(name, 0) + 1
            )
            self._consecutive_successes[name] = 0

        consec_failures = self._consecutive_failures[name]
        consec_successes = self._consecutive_successes[name]

        # Determine current status from history (last recorded)
        last_status = self._last_status(name)

        if last_status == HealthStatus.UNKNOWN:
            # Bootstrap: first probe result determines initial status.
            if success:
                return HealthStatus.HEALTHY
            return HealthStatus.UNHEALTHY

        if last_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED):
            if (
                not success
                and consec_failures >= self._config.unhealthy_threshold
            ):
                return HealthStatus.UNHEALTHY
            if not success:
                # Partial failures before threshold → degraded
                return HealthStatus.DEGRADED
            return HealthStatus.HEALTHY

        # last_status == UNHEALTHY
        if success and consec_successes >= self._config.healthy_threshold:
            return HealthStatus.HEALTHY
        if success:
            # Recovery in progress but threshold not yet met
            return HealthStatus.DEGRADED
        return HealthStatus.UNHEALTHY

    def _last_status(self, name: str) -> HealthStatus:
        """Return the most recent status for an endpoint, or UNKNOWN."""
        history = self._history.get(name)
        if not history:
            return HealthStatus.UNKNOWN
        return history[-1].status

    def _record(self, name: str, result: HealthCheckResult) -> None:
        """Append a result to the endpoint's history, respecting max_history."""
        if name not in self._history:
            self._history[name] = deque(maxlen=self._config.max_history)
        self._history[name].append(result)

    def __repr__(self) -> str:
        tracked = len(self._history)
        return (
            f"HealthChecker("
            f"tracked_endpoints={tracked}, "
            f"timeout={self._config.timeout_seconds}s"
            f")"
        )
