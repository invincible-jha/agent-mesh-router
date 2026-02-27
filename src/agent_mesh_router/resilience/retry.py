"""RetryPolicy — retry failed callables with exponential backoff.

The policy retries a callable on any exception up to ``max_retries`` times,
sleeping between attempts using an exponential backoff formula with optional
uniform jitter.

Backoff formula
---------------
::

    delay = min(base_delay * exponential_base ** attempt, max_delay)
    if jitter:
        delay = random.uniform(0, delay)

``attempt`` is 0-indexed, so the first retry uses
``base_delay * exponential_base ** 0 == base_delay``.

Example
-------
::

    import httpx
    from agent_mesh_router.resilience.retry import RetryPolicy, RetryConfig

    config = RetryConfig(
        max_retries=3,
        base_delay_seconds=0.5,
        max_delay_seconds=10.0,
        exponential_base=2.0,
        jitter=True,
    )
    policy = RetryPolicy(config)

    # Synchronous
    response = policy.execute(lambda: httpx.get("https://example.com"))

    # Asynchronous
    import asyncio
    response = asyncio.run(
        policy.execute_async(lambda: httpx.AsyncClient().get("https://example.com"))
    )
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted.

    Attributes
    ----------
    attempts:
        Total number of attempts made (initial call + retries).
    last_error:
        The final exception that caused exhaustion.
    """

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} attempt(s) failed. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )


@dataclass(frozen=True)
class RetryConfig:
    """Immutable configuration for a ``RetryPolicy``.

    Attributes
    ----------
    max_retries:
        Number of *retry* attempts after the initial call fails.  Total
        attempts = ``max_retries + 1``.  Must be >= 0.
    base_delay_seconds:
        Starting delay in seconds before the first retry.  Must be >= 0.
    max_delay_seconds:
        Upper bound on the computed delay.  Must be >= ``base_delay_seconds``.
    exponential_base:
        Multiplier applied to ``base_delay`` on each successive attempt.
        Must be >= 1.0.  Defaults to 2.0 (classic exponential backoff).
    jitter:
        When True, each computed delay is replaced with a uniform random
        value in ``[0, delay]`` to spread retry bursts.  Defaults to True.
    """

    max_retries: int
    base_delay_seconds: float
    max_delay_seconds: float
    exponential_base: float = field(default=2.0)
    jitter: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}."
            )
        if self.base_delay_seconds < 0:
            raise ValueError(
                f"base_delay_seconds must be >= 0, got {self.base_delay_seconds}."
            )
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                f"max_delay_seconds ({self.max_delay_seconds}) must be >= "
                f"base_delay_seconds ({self.base_delay_seconds})."
            )
        if self.exponential_base < 1.0:
            raise ValueError(
                f"exponential_base must be >= 1.0, got {self.exponential_base}."
            )


class RetryPolicy:
    """Execute callables with configurable retry and backoff behaviour.

    Parameters
    ----------
    config:
        Immutable retry configuration.
    """

    def __init__(self, config: RetryConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, fn: Callable[[], _T]) -> _T:
        """Execute ``fn`` with retry on failure.

        Parameters
        ----------
        fn:
            Zero-argument callable to execute.

        Returns
        -------
        _T
            The return value of ``fn`` on the first successful attempt.

        Raises
        ------
        RetryExhaustedError
            If every attempt raises an exception.
        """
        last_exc: BaseException | None = None
        total_attempts = self._config.max_retries + 1

        for attempt in range(total_attempts):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                remaining = total_attempts - attempt - 1
                logger.debug(
                    "RetryPolicy: attempt %d/%d failed with %s: %s. "
                    "%d attempt(s) remaining.",
                    attempt + 1,
                    total_attempts,
                    type(exc).__name__,
                    exc,
                    remaining,
                )
                if remaining > 0:
                    delay = self._compute_delay(attempt)
                    logger.debug(
                        "RetryPolicy: sleeping %.3f s before next attempt.", delay
                    )
                    time.sleep(delay)

        # Unreachable without last_exc being set, but mypy needs the guard.
        raise RetryExhaustedError(
            attempts=total_attempts,
            last_error=last_exc or RuntimeError("Unknown error"),
        )

    async def execute_async(
        self, fn: Callable[[], Awaitable[_T]]
    ) -> _T:
        """Execute an async callable ``fn`` with retry on failure.

        Parameters
        ----------
        fn:
            Zero-argument callable that returns an awaitable.

        Returns
        -------
        _T
            The return value of the first successful await.

        Raises
        ------
        RetryExhaustedError
            If every attempt raises an exception.
        """
        last_exc: BaseException | None = None
        total_attempts = self._config.max_retries + 1

        for attempt in range(total_attempts):
            try:
                return await fn()
            except Exception as exc:
                last_exc = exc
                remaining = total_attempts - attempt - 1
                logger.debug(
                    "RetryPolicy(async): attempt %d/%d failed with %s: %s. "
                    "%d attempt(s) remaining.",
                    attempt + 1,
                    total_attempts,
                    type(exc).__name__,
                    exc,
                    remaining,
                )
                if remaining > 0:
                    delay = self._compute_delay(attempt)
                    logger.debug(
                        "RetryPolicy(async): sleeping %.3f s before next attempt.",
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise RetryExhaustedError(
            attempts=total_attempts,
            last_error=last_exc or RuntimeError("Unknown error"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_delay(self, attempt: int) -> float:
        """Return the sleep duration for a given attempt index.

        Parameters
        ----------
        attempt:
            Zero-indexed attempt number (0 = delay before second try).

        Returns
        -------
        float
            Seconds to sleep.  Always >= 0.
        """
        raw = min(
            self._config.base_delay_seconds
            * (self._config.exponential_base ** attempt),
            self._config.max_delay_seconds,
        )
        if self._config.jitter:
            return random.uniform(0.0, raw)
        return raw

    def __repr__(self) -> str:
        cfg = self._config
        return (
            f"RetryPolicy("
            f"max_retries={cfg.max_retries}, "
            f"base_delay={cfg.base_delay_seconds}s, "
            f"max_delay={cfg.max_delay_seconds}s, "
            f"base={cfg.exponential_base}, "
            f"jitter={cfg.jitter}"
            f")"
        )
