"""Recovery tester for HALF-OPEN circuit breaker state.

When a circuit breaker transitions from OPEN to HALF_OPEN it allows a
limited number of probe calls through to check whether the downstream
service has recovered.  This module provides a :class:`RecoveryTester`
that encapsulates that probing logic, tracking successes and failures
to determine whether the circuit should close or reopen.

Classes
-------
- RecoveryTestResult   Immutable result summary of a recovery test run.
- RecoveryTester       Runs probe calls and decides circuit fate.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryTestResult:
    """Immutable summary of a single recovery test run.

    Attributes
    ----------
    total_attempts:
        Number of probe calls that were attempted.
    successes:
        Number of probe calls that completed without raising an exception.
    failures:
        Number of probe calls that raised an exception.
    should_close:
        ``True`` when the number of successes meets or exceeds
        :attr:`RecoveryTester.success_threshold`, indicating the circuit
        should transition to CLOSED.
    latency_ms:
        Total elapsed wall-clock time for all probe calls, in milliseconds.
    """

    total_attempts: int
    successes: int
    failures: int
    should_close: bool
    latency_ms: float

    def __post_init__(self) -> None:
        if self.total_attempts < 0:
            raise ValueError("total_attempts must be >= 0.")
        if self.successes < 0:
            raise ValueError("successes must be >= 0.")
        if self.failures < 0:
            raise ValueError("failures must be >= 0.")
        if self.successes + self.failures > self.total_attempts:
            raise ValueError(
                f"successes ({self.successes}) + failures ({self.failures}) "
                f"exceeds total_attempts ({self.total_attempts})."
            )
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0.")

    @property
    def success_rate(self) -> float:
        """Fraction of attempts that succeeded (0.0–1.0).

        Returns 0.0 when total_attempts is zero.
        """
        if self.total_attempts == 0:
            return 0.0
        return self.successes / self.total_attempts


# ---------------------------------------------------------------------------
# Recovery tester
# ---------------------------------------------------------------------------


class RecoveryTester:
    """Tests service recovery during the HALF-OPEN circuit breaker state.

    The tester repeatedly calls *service_fn* (up to ``max_test_requests``
    times) and counts successes and failures.  It stops early once the
    success threshold is reached or once a failure occurs (fail-fast).

    Parameters
    ----------
    max_test_requests:
        Maximum number of probe calls to make.  Must be >= 1.
    success_threshold:
        Number of consecutive successes required to recommend closing the
        circuit (``should_close=True``).  Must be >= 1 and <=
        ``max_test_requests``.
    stop_on_first_failure:
        When ``True`` (default), stop probing as soon as a failure is
        detected — mirroring standard circuit breaker HALF_OPEN semantics
        where any failure reopens the circuit immediately.
    """

    def __init__(
        self,
        max_test_requests: int = 3,
        success_threshold: int = 2,
        stop_on_first_failure: bool = True,
    ) -> None:
        if max_test_requests < 1:
            raise ValueError(f"max_test_requests must be >= 1, got {max_test_requests}.")
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be >= 1, got {success_threshold}.")
        if success_threshold > max_test_requests:
            raise ValueError(
                f"success_threshold ({success_threshold}) must be <= "
                f"max_test_requests ({max_test_requests})."
            )
        self._max_test_requests = max_test_requests
        self._success_threshold = success_threshold
        self._stop_on_first_failure = stop_on_first_failure

    @property
    def max_test_requests(self) -> int:
        """Maximum number of probe calls per recovery test."""
        return self._max_test_requests

    @property
    def success_threshold(self) -> int:
        """Successes needed before recommending circuit closure."""
        return self._success_threshold

    def test_recovery(
        self,
        service_fn: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> RecoveryTestResult:
        """Probe *service_fn* to determine if recovery has occurred.

        Parameters
        ----------
        service_fn:
            Callable representing the downstream service.  Called with
            ``*args`` and ``**kwargs`` for each probe attempt.
        *args:
            Positional arguments forwarded to *service_fn*.
        **kwargs:
            Keyword arguments forwarded to *service_fn*.

        Returns
        -------
        RecoveryTestResult
            Immutable summary including attempt counts and the
            ``should_close`` recommendation.
        """
        successes = 0
        failures = 0
        start = time.monotonic()

        for attempt in range(self._max_test_requests):
            try:
                service_fn(*args, **kwargs)
                successes += 1
                logger.debug(
                    "RecoveryTester: probe %d/%d succeeded (total_successes=%d).",
                    attempt + 1,
                    self._max_test_requests,
                    successes,
                )
                if successes >= self._success_threshold:
                    # Threshold met — no need to continue probing.
                    break
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.debug(
                    "RecoveryTester: probe %d/%d failed: %s.",
                    attempt + 1,
                    self._max_test_requests,
                    exc,
                )
                if self._stop_on_first_failure:
                    break

        elapsed_ms = (time.monotonic() - start) * 1000.0
        total_attempts = successes + failures
        should_close = successes >= self._success_threshold

        result = RecoveryTestResult(
            total_attempts=total_attempts,
            successes=successes,
            failures=failures,
            should_close=should_close,
            latency_ms=elapsed_ms,
        )
        logger.info(
            "RecoveryTester: %d/%d probes succeeded — should_close=%s (%.2f ms).",
            successes,
            total_attempts,
            should_close,
            elapsed_ms,
        )
        return result

    def __repr__(self) -> str:
        return (
            f"RecoveryTester("
            f"max_test_requests={self._max_test_requests}, "
            f"success_threshold={self._success_threshold}, "
            f"stop_on_first_failure={self._stop_on_first_failure}"
            f")"
        )


__all__ = [
    "RecoveryTestResult",
    "RecoveryTester",
]
