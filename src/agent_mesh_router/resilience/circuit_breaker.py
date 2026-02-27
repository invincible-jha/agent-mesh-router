"""CircuitBreaker — protect downstream calls from cascading failures.

The circuit breaker pattern wraps a callable and tracks its success/failure
rate.  When failures exceed a threshold the circuit *opens*, rejecting all
calls immediately rather than waiting for a slow or broken dependency.
After a configurable recovery timeout the circuit moves to *half-open*,
allowing a limited number of probe calls through.  If those probes succeed
the circuit *closes* again; a single failure returns it to *open*.

State machine
-------------

::

    CLOSED ──(failures >= threshold)──► OPEN
       ▲                                  │
       │                          (timeout elapsed)
       │                                  ▼
       └──(successes >= threshold)── HALF_OPEN
                                          │
                                   (any failure)
                                          │
                                          ▼
                                        OPEN

Example
-------
::

    from agent_mesh_router.resilience.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerOpenError,
        CircuitState,
    )

    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=2,
        success_threshold=2,
    )
    cb = CircuitBreaker(config, name="payments-api")

    try:
        result = cb.execute(lambda: call_external_api())
    except CircuitBreakerOpenError:
        result = use_fallback()
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class CircuitState(str, Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is open.

    Attributes
    ----------
    circuit_name:
        The name of the circuit breaker that rejected the call.
    state:
        The state at the time of rejection (always ``OPEN``).
    """

    def __init__(self, circuit_name: str) -> None:
        self.circuit_name = circuit_name
        super().__init__(
            f"Circuit breaker '{circuit_name}' is OPEN — call rejected."
        )


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Immutable configuration for a ``CircuitBreaker``.

    Attributes
    ----------
    failure_threshold:
        Number of consecutive failures in CLOSED state required to open
        the circuit.  Must be >= 1.
    recovery_timeout_seconds:
        Seconds the circuit stays OPEN before transitioning to HALF_OPEN.
        Must be > 0.
    half_open_max_calls:
        Maximum number of probe calls allowed while in HALF_OPEN.  Any
        further calls are rejected until the circuit resolves.  Must be >= 1.
    success_threshold:
        Number of consecutive successes in HALF_OPEN required to close
        the circuit.  Must be >= 1 and <= ``half_open_max_calls``.
    """

    failure_threshold: int
    recovery_timeout_seconds: float
    half_open_max_calls: int
    success_threshold: int

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError(
                f"failure_threshold must be >= 1, got {self.failure_threshold}."
            )
        if self.recovery_timeout_seconds <= 0:
            raise ValueError(
                f"recovery_timeout_seconds must be > 0, "
                f"got {self.recovery_timeout_seconds}."
            )
        if self.half_open_max_calls < 1:
            raise ValueError(
                f"half_open_max_calls must be >= 1, "
                f"got {self.half_open_max_calls}."
            )
        if self.success_threshold < 1:
            raise ValueError(
                f"success_threshold must be >= 1, got {self.success_threshold}."
            )
        if self.success_threshold > self.half_open_max_calls:
            raise ValueError(
                f"success_threshold ({self.success_threshold}) must be <= "
                f"half_open_max_calls ({self.half_open_max_calls})."
            )


class CircuitBreaker:
    """Wrap callables with circuit-breaker protection.

    All state mutations are thread-safe.

    Parameters
    ----------
    config:
        Immutable configuration object.
    name:
        Optional human-readable name used in logs and error messages.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        *,
        name: str = "unnamed",
    ) -> None:
        self._config = config
        self._name = name
        self._lock = threading.RLock()

        # State
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_call_count: int = 0
        self._opened_at: float | None = None

        # Lifetime metrics
        self._total_calls: int = 0
        self._total_successes: int = 0
        self._total_failures: int = 0
        self._total_rejected: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (thread-safe snapshot)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def name(self) -> str:
        """Human-readable circuit breaker name."""
        return self._name

    @property
    def failure_count(self) -> int:
        """Consecutive failure count in the current CLOSED window."""
        with self._lock:
            return self._failure_count

    @property
    def total_calls(self) -> int:
        """Total calls attempted (not counting rejected calls)."""
        return self._total_calls

    @property
    def total_rejected(self) -> int:
        """Total calls rejected because the circuit was open."""
        return self._total_rejected

    def execute(self, fn: Callable[[], _T]) -> _T:
        """Execute ``fn`` if the circuit allows, tracking success/failure.

        Parameters
        ----------
        fn:
            Zero-argument callable to execute.

        Returns
        -------
        _T
            The return value of ``fn``.

        Raises
        ------
        CircuitBreakerOpenError
            If the circuit is open or the HALF_OPEN probe quota is full.
        Exception
            Any exception raised by ``fn`` is re-raised after recording
            the failure.
        """
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                self._total_rejected += 1
                raise CircuitBreakerOpenError(self._name)

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_call_count >= self._config.half_open_max_calls:
                    self._total_rejected += 1
                    raise CircuitBreakerOpenError(self._name)
                self._half_open_call_count += 1

            self._total_calls += 1

        try:
            result = fn()
        except Exception as exc:
            self.record_failure()
            raise exc
        else:
            self.record_success()
            return result

    def record_success(self) -> None:
        """Manually record a successful operation.

        Use when integrating with code that does not call ``execute``.
        """
        with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.CLOSED:
                # Reset consecutive failure window on any success
                self._failure_count = 0

            elif self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self) -> None:
        """Manually record a failed operation.

        Use when integrating with code that does not call ``execute``.
        """
        with self._lock:
            self._total_failures += 1

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN reopens the circuit immediately
                self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Force the circuit to CLOSED and clear all counters.

        Useful in tests or administrative override scenarios.
        """
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """Transition OPEN → HALF_OPEN when the recovery timeout has elapsed.

        Must be called with ``self._lock`` held.
        """
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at
            >= self._config.recovery_timeout_seconds
        ):
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Perform a state transition, updating ancillary counters.

        Must be called with ``self._lock`` held.
        """
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._failure_count = 0
            self._success_count = 0
            self._half_open_call_count = 0

        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_call_count = 0

        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._failure_count = 0
            self._success_count = 0
            self._half_open_call_count = 0

        logger.info(
            "CircuitBreaker[%s]: %s → %s.",
            self._name,
            old_state.value,
            new_state.value,
        )

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker("
            f"name={self._name!r}, "
            f"state={self._state.value}, "
            f"failures={self._failure_count}, "
            f"calls={self._total_calls}"
            f")"
        )
