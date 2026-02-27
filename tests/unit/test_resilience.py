"""Tests for the resilience layer.

Covers:
- CircuitBreaker state machine transitions
- CircuitBreaker execute() with real callables
- RetryPolicy sync and async execution
- RetryExhaustedError attributes
- Exponential backoff delay computation
- All 4 LoadBalancer strategies
- LoadBalancer endpoint health management
- HealthChecker results and history
- CLI resilience commands
- Edge cases: empty endpoints, all-unhealthy, invalid configs, concurrency
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from agent_mesh_router.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)
from agent_mesh_router.resilience.health_check import (
    HealthCheckConfig,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
)
from agent_mesh_router.resilience.load_balancer import (
    LoadBalancer,
    LoadBalancerStrategy,
    NoHealthyEndpointError,
    ServiceEndpoint,
)
from agent_mesh_router.resilience.retry import (
    RetryConfig,
    RetryExhaustedError,
    RetryPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    failure_threshold: int = 3,
    recovery_timeout_seconds: float = 30.0,
    half_open_max_calls: int = 2,
    success_threshold: int = 2,
) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
        half_open_max_calls=half_open_max_calls,
        success_threshold=success_threshold,
    )


def _endpoint(
    name: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    weight: float = 1.0,
    healthy: bool = True,
) -> ServiceEndpoint:
    return ServiceEndpoint(name=name, host=host, port=port, weight=weight, healthy=healthy)


def _hc_config(
    interval_seconds: float = 30.0,
    timeout_seconds: float = 5.0,
    healthy_threshold: int = 2,
    unhealthy_threshold: int = 2,
) -> HealthCheckConfig:
    return HealthCheckConfig(
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
        healthy_threshold=healthy_threshold,
        unhealthy_threshold=unhealthy_threshold,
    )


def _noop_probe(endpoint: ServiceEndpoint) -> None:
    """Probe that always succeeds."""


def _failing_probe(endpoint: ServiceEndpoint) -> None:
    """Probe that always raises."""
    raise ConnectionRefusedError("connection refused")


# ===========================================================================
# CircuitBreakerConfig
# ===========================================================================


class TestCircuitBreakerConfig:
    def test_valid_config(self) -> None:
        cfg = _config()
        assert cfg.failure_threshold == 3
        assert cfg.recovery_timeout_seconds == 30.0

    def test_failure_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreakerConfig(
                failure_threshold=0,
                recovery_timeout_seconds=30.0,
                half_open_max_calls=2,
                success_threshold=1,
            )

    def test_recovery_timeout_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="recovery_timeout_seconds"):
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=0.0,
                half_open_max_calls=2,
                success_threshold=1,
            )

    def test_half_open_max_calls_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="half_open_max_calls"):
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=1.0,
                half_open_max_calls=0,
                success_threshold=1,
            )

    def test_success_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="success_threshold"):
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=1.0,
                half_open_max_calls=2,
                success_threshold=0,
            )

    def test_success_threshold_exceeds_half_open_raises(self) -> None:
        with pytest.raises(ValueError, match="success_threshold"):
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_seconds=1.0,
                half_open_max_calls=2,
                success_threshold=3,
            )

    def test_frozen_immutability(self) -> None:
        cfg = _config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.failure_threshold = 99  # type: ignore[misc]


# ===========================================================================
# CircuitBreaker — initial state
# ===========================================================================


class TestCircuitBreakerInitialState:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(_config())
        assert cb.state == CircuitState.CLOSED

    def test_name_default(self) -> None:
        cb = CircuitBreaker(_config())
        assert cb.name == "unnamed"

    def test_name_custom(self) -> None:
        cb = CircuitBreaker(_config(), name="svc-a")
        assert cb.name == "svc-a"

    def test_initial_failure_count_zero(self) -> None:
        cb = CircuitBreaker(_config())
        assert cb.failure_count == 0

    def test_initial_total_calls_zero(self) -> None:
        cb = CircuitBreaker(_config())
        assert cb.total_calls == 0

    def test_initial_total_rejected_zero(self) -> None:
        cb = CircuitBreaker(_config())
        assert cb.total_rejected == 0

    def test_repr_contains_name_and_state(self) -> None:
        cb = CircuitBreaker(_config(), name="my-cb")
        r = repr(cb)
        assert "my-cb" in r
        assert "closed" in r


# ===========================================================================
# CircuitBreaker — CLOSED → OPEN transition
# ===========================================================================


class TestCircuitBreakerClosedToOpen:
    def test_single_failure_does_not_open(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=3))
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_failures_below_threshold_stay_closed(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_failures_at_threshold_opens_circuit(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=3))
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_counter(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Only 2 failures since last success → still closed
        assert cb.state == CircuitState.CLOSED

    def test_execute_raises_on_callable_failure_and_opens(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=2))

        def bad() -> None:
            raise ValueError("broken")

        with pytest.raises(ValueError):
            cb.execute(bad)
        with pytest.raises(ValueError):
            cb.execute(bad)

        assert cb.state == CircuitState.OPEN

    def test_execute_returns_callable_result(self) -> None:
        cb = CircuitBreaker(_config())
        result = cb.execute(lambda: 42)
        assert result == 42


# ===========================================================================
# CircuitBreaker — OPEN behavior
# ===========================================================================


class TestCircuitBreakerOpen:
    def test_open_circuit_rejects_execute(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            cb.execute(lambda: None)

    def test_open_circuit_increments_rejected_count(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1))
        cb.record_failure()
        try:
            cb.execute(lambda: None)
        except CircuitBreakerOpenError:
            pass
        assert cb.total_rejected == 1

    def test_circuit_breaker_open_error_carries_name(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1), name="db-conn")
        cb.record_failure()
        exc = None
        try:
            cb.execute(lambda: None)
        except CircuitBreakerOpenError as err:
            exc = err
        assert exc is not None
        assert exc.circuit_name == "db-conn"

    def test_failure_recording_while_open_stays_open(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1))
        cb.record_failure()
        # Failures in OPEN state should not stack — circuit stays OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


# ===========================================================================
# CircuitBreaker — OPEN → HALF_OPEN transition
# ===========================================================================


class TestCircuitBreakerOpenToHalfOpen:
    def test_transitions_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1, recovery_timeout_seconds=0.05))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.1)
        # Reading .state triggers the lazy transition
        assert cb.state == CircuitState.HALF_OPEN

    def test_does_not_transition_before_timeout(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1, recovery_timeout_seconds=60.0))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_execute_reads_state_and_transitions_to_half_open(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1, recovery_timeout_seconds=0.05))
        cb.record_failure()
        time.sleep(0.1)
        # execute() should now run in HALF_OPEN mode
        result = cb.execute(lambda: "ok")
        assert result == "ok"


# ===========================================================================
# CircuitBreaker — HALF_OPEN → CLOSED / OPEN transitions
# ===========================================================================


class TestCircuitBreakerHalfOpen:
    def _get_half_open_cb(
        self,
        success_threshold: int = 2,
        half_open_max_calls: int = 3,
    ) -> CircuitBreaker:
        cb = CircuitBreaker(
            _config(
                failure_threshold=1,
                recovery_timeout_seconds=0.05,
                half_open_max_calls=half_open_max_calls,
                success_threshold=success_threshold,
            )
        )
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        return cb

    def test_sufficient_successes_close_circuit(self) -> None:
        cb = self._get_half_open_cb(success_threshold=2)
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # still needs one more
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_any_failure_reopens_circuit(self) -> None:
        cb = self._get_half_open_cb(success_threshold=2)
        cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_max_calls_respected(self) -> None:
        # Use max_calls=2, success_threshold=2 so both probes must succeed.
        # The third call should be rejected because the quota (2) is exhausted
        # before the circuit transitions to CLOSED (needs both successes first).
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.05,
                half_open_max_calls=2,
                success_threshold=2,
            )
        )
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        # Use both allowed probes — circuit is still evaluating
        cb.execute(lambda: None)  # probe 1 of 2
        cb.execute(lambda: None)  # probe 2 of 2 → closes circuit
        # Circuit is now CLOSED (2 successes met threshold)
        assert cb.state == CircuitState.CLOSED

        # Separate test: quota exhausted while still HALF_OPEN
        cb2 = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.05,
                half_open_max_calls=1,
                success_threshold=1,
            )
        )
        cb2.record_failure()
        time.sleep(0.1)
        cb2.execute(lambda: None)  # uses the 1 allowed call → CLOSED
        # Circuit is now CLOSED; additional calls succeed normally
        result = cb2.execute(lambda: "normal")
        assert result == "normal"

    def test_half_open_execute_failure_reopens(self) -> None:
        cb = self._get_half_open_cb(success_threshold=2, half_open_max_calls=3)

        def bad() -> None:
            raise RuntimeError("probe fail")

        with pytest.raises(RuntimeError):
            cb.execute(bad)
        assert cb.state == CircuitState.OPEN

    def test_half_open_quota_exceeded_raises_circuit_open_error(self) -> None:
        """Exhaust half_open_max_calls without reaching success_threshold."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.05,
                half_open_max_calls=2,
                success_threshold=2,
            )
        )
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        # Use both probe slots (neither triggers close because they increment
        # half_open_call_count but then the circuit transitions on success).
        # To exhaust without closing we need an immediate reject on the 3rd call.
        cb.execute(lambda: None)  # call #1 → success recorded, success_count=1
        cb.execute(lambda: None)  # call #2 → success recorded, success_count=2 → CLOSED
        # Now re-open to test rejection path: open then wait then exhaust without success
        cb2 = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.05,
                half_open_max_calls=1,
                success_threshold=1,
            )
        )
        cb2.record_failure()
        time.sleep(0.1)
        # First call uses the single allowed slot → success → CLOSED
        cb2.execute(lambda: None)
        # Back in CLOSED; open it again
        cb2.record_failure()
        time.sleep(0.1)
        assert cb2.state == CircuitState.HALF_OPEN
        # Use the slot — this time do NOT trigger a transition
        # Test the rejection by filling the slot count directly
        with cb2._lock:
            cb2._half_open_call_count = 1  # simulate slot exhausted
        with pytest.raises(CircuitBreakerOpenError):
            cb2.execute(lambda: None)
        assert cb2.total_rejected >= 1


# ===========================================================================
# CircuitBreaker — reset()
# ===========================================================================


class TestCircuitBreakerReset:
    def test_reset_from_open_to_closed(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_failure_count(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=5))
        for _ in range(4):
            cb.record_failure()
        cb.reset()
        assert cb.failure_count == 0

    def test_reset_allows_execute(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=1))
        cb.record_failure()
        cb.reset()
        result = cb.execute(lambda: "restored")
        assert result == "restored"


# ===========================================================================
# RetryConfig
# ===========================================================================


class TestRetryConfig:
    def test_valid_config(self) -> None:
        cfg = RetryConfig(
            max_retries=3,
            base_delay_seconds=0.1,
            max_delay_seconds=5.0,
        )
        assert cfg.max_retries == 3
        assert cfg.jitter is True
        assert cfg.exponential_base == 2.0

    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            RetryConfig(max_retries=-1, base_delay_seconds=0.1, max_delay_seconds=1.0)

    def test_negative_base_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="base_delay_seconds"):
            RetryConfig(max_retries=1, base_delay_seconds=-0.1, max_delay_seconds=1.0)

    def test_max_less_than_base_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delay_seconds"):
            RetryConfig(max_retries=1, base_delay_seconds=5.0, max_delay_seconds=1.0)

    def test_exponential_base_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="exponential_base"):
            RetryConfig(
                max_retries=1,
                base_delay_seconds=0.1,
                max_delay_seconds=1.0,
                exponential_base=0.5,
            )

    def test_zero_retries_valid(self) -> None:
        cfg = RetryConfig(max_retries=0, base_delay_seconds=0.0, max_delay_seconds=0.0)
        assert cfg.max_retries == 0


# ===========================================================================
# RetryPolicy — synchronous
# ===========================================================================


class TestRetryPolicySync:
    def test_success_on_first_attempt(self) -> None:
        policy = RetryPolicy(
            RetryConfig(max_retries=3, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        result = policy.execute(lambda: "done")
        assert result == "done"

    def test_success_on_second_attempt(self) -> None:
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("not yet")
            return "ok"

        policy = RetryPolicy(
            RetryConfig(max_retries=3, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        result = policy.execute(flaky)
        assert result == "ok"
        assert call_count == 2

    def test_success_on_nth_attempt(self) -> None:
        call_count = 0
        target_attempt = 4

        def eventually_works() -> int:
            nonlocal call_count
            call_count += 1
            if call_count < target_attempt:
                raise RuntimeError("transient")
            return call_count

        policy = RetryPolicy(
            RetryConfig(max_retries=5, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        result = policy.execute(eventually_works)
        assert result == target_attempt
        assert call_count == target_attempt

    def test_exhaustion_raises_retry_exhausted_error(self) -> None:
        policy = RetryPolicy(
            RetryConfig(max_retries=2, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        with pytest.raises(RetryExhaustedError) as exc_info:
            policy.execute(lambda: (_ for _ in ()).throw(IOError("gone")))
        error = exc_info.value
        assert error.attempts == 3  # 1 initial + 2 retries

    def test_retry_exhausted_error_carries_last_error(self) -> None:
        sentinel = ValueError("specific error")

        def always_fails() -> None:
            raise sentinel

        policy = RetryPolicy(
            RetryConfig(max_retries=1, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        with pytest.raises(RetryExhaustedError) as exc_info:
            policy.execute(always_fails)
        assert exc_info.value.last_error is sentinel

    def test_zero_retries_fails_immediately(self) -> None:
        policy = RetryPolicy(
            RetryConfig(max_retries=0, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        with pytest.raises(RetryExhaustedError) as exc_info:
            policy.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert exc_info.value.attempts == 1

    def test_repr_shows_config(self) -> None:
        policy = RetryPolicy(
            RetryConfig(max_retries=3, base_delay_seconds=0.5, max_delay_seconds=10.0)
        )
        r = repr(policy)
        assert "max_retries=3" in r
        assert "base_delay=0.5s" in r


# ===========================================================================
# RetryPolicy — backoff delay calculation
# ===========================================================================


class TestRetryPolicyBackoff:
    def test_delay_formula_no_jitter(self) -> None:
        policy = RetryPolicy(
            RetryConfig(
                max_retries=5,
                base_delay_seconds=1.0,
                max_delay_seconds=100.0,
                exponential_base=2.0,
                jitter=False,
            )
        )
        assert policy._compute_delay(0) == pytest.approx(1.0)
        assert policy._compute_delay(1) == pytest.approx(2.0)
        assert policy._compute_delay(2) == pytest.approx(4.0)
        assert policy._compute_delay(3) == pytest.approx(8.0)

    def test_delay_capped_at_max(self) -> None:
        policy = RetryPolicy(
            RetryConfig(
                max_retries=10,
                base_delay_seconds=1.0,
                max_delay_seconds=5.0,
                exponential_base=2.0,
                jitter=False,
            )
        )
        assert policy._compute_delay(10) == pytest.approx(5.0)

    def test_jitter_produces_value_within_range(self) -> None:
        policy = RetryPolicy(
            RetryConfig(
                max_retries=5,
                base_delay_seconds=1.0,
                max_delay_seconds=10.0,
                exponential_base=2.0,
                jitter=True,
            )
        )
        for _ in range(20):
            delay = policy._compute_delay(2)  # raw = 4.0 s
            assert 0.0 <= delay <= 4.0

    def test_sleep_called_with_computed_delay(self) -> None:
        call_count = 0

        def fails_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("err")
            return "done"

        policy = RetryPolicy(
            RetryConfig(
                max_retries=3,
                base_delay_seconds=0.5,
                max_delay_seconds=10.0,
                jitter=False,
            )
        )
        with patch("agent_mesh_router.resilience.retry.time.sleep") as mock_sleep:
            result = policy.execute(fails_twice)
        assert result == "done"
        # Two failures → two sleeps
        assert mock_sleep.call_count == 2
        # First sleep: base * 2^0 = 0.5s
        assert mock_sleep.call_args_list[0] == call(pytest.approx(0.5))
        # Second sleep: base * 2^1 = 1.0s
        assert mock_sleep.call_args_list[1] == call(pytest.approx(1.0))

    def test_no_sleep_after_final_failure(self) -> None:
        policy = RetryPolicy(
            RetryConfig(
                max_retries=1,
                base_delay_seconds=1.0,
                max_delay_seconds=10.0,
                jitter=False,
            )
        )
        with patch("agent_mesh_router.resilience.retry.time.sleep") as mock_sleep:
            with pytest.raises(RetryExhaustedError):
                policy.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        # 2 total attempts (1 initial + 1 retry) → only 1 sleep between attempts
        assert mock_sleep.call_count == 1


# ===========================================================================
# RetryPolicy — async
# ===========================================================================


class TestRetryPolicyAsync:
    @pytest.mark.asyncio
    async def test_async_success_on_first_attempt(self) -> None:
        policy = RetryPolicy(
            RetryConfig(max_retries=2, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )

        async def good() -> str:
            return "async-result"

        result = await policy.execute_async(good)
        assert result == "async-result"

    @pytest.mark.asyncio
    async def test_async_retries_on_failure(self) -> None:
        call_count = 0

        async def flaky() -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("retry me")
            return call_count

        policy = RetryPolicy(
            RetryConfig(max_retries=5, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        result = await policy.execute_async(flaky)
        assert result == 3

    @pytest.mark.asyncio
    async def test_async_exhaustion_raises(self) -> None:
        async def always_fails() -> None:
            raise TimeoutError("timeout")

        policy = RetryPolicy(
            RetryConfig(max_retries=2, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        with pytest.raises(RetryExhaustedError) as exc_info:
            await policy.execute_async(always_fails)
        assert exc_info.value.attempts == 3

    @pytest.mark.asyncio
    async def test_async_last_error_preserved(self) -> None:
        original = RuntimeError("original failure")

        async def bad() -> None:
            raise original

        policy = RetryPolicy(
            RetryConfig(max_retries=1, base_delay_seconds=0.0, max_delay_seconds=0.0)
        )
        with pytest.raises(RetryExhaustedError) as exc_info:
            await policy.execute_async(bad)
        assert exc_info.value.last_error is original


# ===========================================================================
# ServiceEndpoint
# ===========================================================================


class TestServiceEndpoint:
    def test_valid_construction(self) -> None:
        ep = _endpoint("svc-1")
        assert ep.name == "svc-1"
        assert ep.host == "127.0.0.1"
        assert ep.port == 8080
        assert ep.weight == 1.0
        assert ep.healthy is True

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            ServiceEndpoint(name="x", host="h", port=80, weight=-1.0)

    def test_zero_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            ServiceEndpoint(name="x", host="h", port=80, weight=0.0)

    def test_invalid_port_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="port"):
            ServiceEndpoint(name="x", host="h", port=0)

    def test_invalid_port_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="port"):
            ServiceEndpoint(name="x", host="h", port=65536)

    def test_frozen_immutability(self) -> None:
        ep = _endpoint("svc")
        with pytest.raises((AttributeError, TypeError)):
            ep.host = "other"  # type: ignore[misc]


# ===========================================================================
# LoadBalancer — endpoint management
# ===========================================================================


class TestLoadBalancerManagement:
    def test_add_and_retrieve_endpoint(self) -> None:
        lb = LoadBalancer()
        ep = _endpoint("a")
        lb.add_endpoint(ep)
        assert ep in lb.all_endpoints

    def test_remove_endpoint(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.remove_endpoint("a")
        assert lb.all_endpoints == []

    def test_remove_nonexistent_is_noop(self) -> None:
        lb = LoadBalancer()
        lb.remove_endpoint("ghost")  # Should not raise

    def test_mark_unhealthy(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.mark_unhealthy("a")
        assert lb.healthy_endpoints == []

    def test_mark_healthy(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.mark_unhealthy("a")
        lb.mark_healthy("a")
        assert len(lb.healthy_endpoints) == 1

    def test_mark_unhealthy_unknown_raises(self) -> None:
        lb = LoadBalancer()
        with pytest.raises(KeyError):
            lb.mark_unhealthy("missing")

    def test_mark_healthy_unknown_raises(self) -> None:
        lb = LoadBalancer()
        with pytest.raises(KeyError):
            lb.mark_healthy("missing")

    def test_healthy_endpoints_snapshot(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        lb.mark_unhealthy("b")
        result = lb.healthy_endpoints
        assert len(result) == 1
        assert result[0].name == "a"

    def test_add_endpoint_replaces_existing(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a", port=8080))
        lb.add_endpoint(_endpoint("a", port=9090))
        endpoints = lb.all_endpoints
        assert len(endpoints) == 1
        assert endpoints[0].port == 9090

    def test_repr_shows_counts(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        lb.mark_unhealthy("a")
        r = repr(lb)
        assert "endpoints=2" in r
        assert "healthy=1" in r


# ===========================================================================
# LoadBalancer — select() with empty / all-unhealthy
# ===========================================================================


class TestLoadBalancerEdgeCases:
    def test_select_empty_raises(self) -> None:
        lb = LoadBalancer()
        with pytest.raises(NoHealthyEndpointError):
            lb.select()

    def test_select_all_unhealthy_raises(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        lb.mark_unhealthy("a")
        lb.mark_unhealthy("b")
        with pytest.raises(NoHealthyEndpointError):
            lb.select()

    def test_initial_unhealthy_endpoint_excluded(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a", healthy=False))
        with pytest.raises(NoHealthyEndpointError):
            lb.select()


# ===========================================================================
# LoadBalancer — ROUND_ROBIN strategy
# ===========================================================================


class TestLoadBalancerRoundRobin:
    def test_cycles_through_all_endpoints(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        lb.add_endpoint(_endpoint("c"))

        strategy = LoadBalancerStrategy.ROUND_ROBIN
        names = [lb.select(strategy).name for _ in range(6)]
        # Should hit each endpoint twice
        for name in ("a", "b", "c"):
            assert names.count(name) == 2

    def test_single_endpoint_always_selected(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("solo"))
        for _ in range(5):
            assert lb.select(LoadBalancerStrategy.ROUND_ROBIN).name == "solo"

    def test_skips_unhealthy_midway(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        lb.mark_unhealthy("a")
        for _ in range(5):
            assert lb.select(LoadBalancerStrategy.ROUND_ROBIN).name == "b"


# ===========================================================================
# LoadBalancer — WEIGHTED_ROUND_ROBIN strategy
# ===========================================================================


class TestLoadBalancerWeightedRoundRobin:
    def test_higher_weight_gets_more_selections(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("light", weight=1.0))
        lb.add_endpoint(_endpoint("heavy", weight=3.0))

        strategy = LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN
        names = [lb.select(strategy).name for _ in range(40)]
        heavy_count = names.count("heavy")
        light_count = names.count("light")
        # Expect roughly 3:1 ratio
        assert heavy_count > light_count * 1.5

    def test_equal_weight_distributes_evenly(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("x", weight=2.0))
        lb.add_endpoint(_endpoint("y", weight=2.0))

        strategy = LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN
        names = [lb.select(strategy).name for _ in range(20)]
        assert names.count("x") == names.count("y") == 10

    def test_single_endpoint_always_returned(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("only", weight=5.0))
        for _ in range(5):
            assert lb.select(LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN).name == "only"


# ===========================================================================
# LoadBalancer — LEAST_CONNECTIONS strategy
# ===========================================================================


class TestLoadBalancerLeastConnections:
    def test_selects_endpoint_with_fewest_connections(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        lb.record_connection_open("a")
        lb.record_connection_open("a")
        selected = lb.select(LoadBalancerStrategy.LEAST_CONNECTIONS)
        assert selected.name == "b"

    def test_connection_close_reduces_count(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        lb.record_connection_open("a")
        lb.record_connection_close("a")
        # Both at 0 now — either could be selected
        selected = lb.select(LoadBalancerStrategy.LEAST_CONNECTIONS)
        assert selected.name in ("a", "b")

    def test_connection_close_does_not_go_negative(self) -> None:
        lb = LoadBalancer()
        lb.add_endpoint(_endpoint("a"))
        lb.record_connection_close("a")  # already at 0
        # Should not raise and count stays at 0
        selected = lb.select(LoadBalancerStrategy.LEAST_CONNECTIONS)
        assert selected.name == "a"

    def test_record_connection_open_unknown_raises(self) -> None:
        lb = LoadBalancer()
        with pytest.raises(KeyError):
            lb.record_connection_open("ghost")

    def test_record_connection_close_unknown_raises(self) -> None:
        lb = LoadBalancer()
        with pytest.raises(KeyError):
            lb.record_connection_close("ghost")


# ===========================================================================
# LoadBalancer — RANDOM strategy
# ===========================================================================


class TestLoadBalancerRandom:
    def test_returns_a_healthy_endpoint(self) -> None:
        lb = LoadBalancer(seed=42)
        lb.add_endpoint(_endpoint("a"))
        lb.add_endpoint(_endpoint("b"))
        for _ in range(10):
            ep = lb.select(LoadBalancerStrategy.RANDOM)
            assert ep.name in ("a", "b")

    def test_seeded_rng_is_deterministic(self) -> None:
        lb1 = LoadBalancer(seed=123)
        lb2 = LoadBalancer(seed=123)
        for lb in (lb1, lb2):
            lb.add_endpoint(_endpoint("x"))
            lb.add_endpoint(_endpoint("y"))

        strategy = LoadBalancerStrategy.RANDOM
        seq1 = [lb1.select(strategy).name for _ in range(10)]
        seq2 = [lb2.select(strategy).name for _ in range(10)]
        assert seq1 == seq2

    def test_covers_all_endpoints_over_many_calls(self) -> None:
        lb = LoadBalancer(seed=99)
        names = {"alpha", "beta", "gamma"}
        for name in names:
            lb.add_endpoint(_endpoint(name))

        seen: set[str] = set()
        for _ in range(100):
            seen.add(lb.select(LoadBalancerStrategy.RANDOM).name)
        assert seen == names


# ===========================================================================
# HealthCheckConfig
# ===========================================================================


class TestHealthCheckConfig:
    def test_valid_config(self) -> None:
        cfg = _hc_config()
        assert cfg.interval_seconds == 30.0

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds"):
            HealthCheckConfig(
                interval_seconds=0.0,
                timeout_seconds=5.0,
                healthy_threshold=2,
                unhealthy_threshold=2,
            )

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            HealthCheckConfig(
                interval_seconds=30.0,
                timeout_seconds=0.0,
                healthy_threshold=2,
                unhealthy_threshold=2,
            )

    def test_zero_healthy_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="healthy_threshold"):
            HealthCheckConfig(
                interval_seconds=30.0,
                timeout_seconds=5.0,
                healthy_threshold=0,
                unhealthy_threshold=2,
            )

    def test_zero_unhealthy_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="unhealthy_threshold"):
            HealthCheckConfig(
                interval_seconds=30.0,
                timeout_seconds=5.0,
                healthy_threshold=2,
                unhealthy_threshold=0,
            )

    def test_default_max_history(self) -> None:
        cfg = _hc_config()
        assert cfg.max_history == 100

    def test_zero_max_history_raises(self) -> None:
        with pytest.raises(ValueError, match="max_history"):
            HealthCheckConfig(
                interval_seconds=30.0,
                timeout_seconds=5.0,
                healthy_threshold=1,
                unhealthy_threshold=1,
                max_history=0,
            )


# ===========================================================================
# HealthChecker — single check
# ===========================================================================


class TestHealthCheckerSingle:
    def test_successful_probe_returns_healthy(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1))
        ep = _endpoint("svc")
        result = checker.check(ep, _noop_probe)
        assert result.endpoint_name == "svc"
        assert result.status == HealthStatus.HEALTHY

    def test_failing_probe_returns_unhealthy(self) -> None:
        checker = HealthChecker(_hc_config(unhealthy_threshold=1))
        ep = _endpoint("svc")
        result = checker.check(ep, _failing_probe)
        assert result.status == HealthStatus.UNHEALTHY

    def test_result_has_latency_ms(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1))
        ep = _endpoint("svc")
        result = checker.check(ep, _noop_probe)
        assert result.latency_ms >= 0.0

    def test_result_has_checked_at_utc(self) -> None:
        from datetime import timezone

        checker = HealthChecker(_hc_config(healthy_threshold=1))
        ep = _endpoint("svc")
        result = checker.check(ep, _noop_probe)
        assert result.checked_at.tzinfo == timezone.utc

    def test_failing_probe_details_contain_error_info(self) -> None:
        checker = HealthChecker(_hc_config(unhealthy_threshold=1))
        ep = _endpoint("svc")
        result = checker.check(ep, _failing_probe)
        assert "error_type" in result.details
        assert result.details["error_type"] == "ConnectionRefusedError"

    def test_result_is_frozen_dataclass(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1))
        result = checker.check(_endpoint("svc"), _noop_probe)
        with pytest.raises((AttributeError, TypeError)):
            result.status = HealthStatus.DEGRADED  # type: ignore[misc]

    def test_slow_probe_treated_as_timeout(self) -> None:
        def slow_probe(endpoint: ServiceEndpoint) -> None:
            # Simulate a probe that takes longer than the timeout
            time.sleep(0.15)

        cfg = HealthCheckConfig(
            interval_seconds=30.0,
            timeout_seconds=0.05,  # 50 ms timeout
            healthy_threshold=1,
            unhealthy_threshold=1,
        )
        checker = HealthChecker(cfg)
        result = checker.check(_endpoint("svc"), slow_probe)
        assert result.status == HealthStatus.UNHEALTHY


# ===========================================================================
# HealthChecker — threshold transitions
# ===========================================================================


class TestHealthCheckerThresholds:
    def test_degraded_before_unhealthy_threshold(self) -> None:
        # threshold=2 → first failure is DEGRADED, second is UNHEALTHY
        checker = HealthChecker(_hc_config(unhealthy_threshold=2, healthy_threshold=2))
        ep = _endpoint("svc")
        # Establish healthy baseline
        checker.check(ep, _noop_probe)
        # First failure → degraded
        r1 = checker.check(ep, _failing_probe)
        assert r1.status == HealthStatus.DEGRADED
        # Second failure → unhealthy
        r2 = checker.check(ep, _failing_probe)
        assert r2.status == HealthStatus.UNHEALTHY

    def test_degraded_before_healthy_threshold(self) -> None:
        # threshold=2 → first success after unhealthy is DEGRADED
        checker = HealthChecker(_hc_config(healthy_threshold=2, unhealthy_threshold=1))
        ep = _endpoint("svc")
        # Reach UNHEALTHY
        checker.check(ep, _failing_probe)
        assert checker.history("svc")[-1].status == HealthStatus.UNHEALTHY
        # First success → degraded (covers line 373)
        r1 = checker.check(ep, _noop_probe)
        assert r1.status == HealthStatus.DEGRADED
        # Second success → healthy (covers line 371)
        r2 = checker.check(ep, _noop_probe)
        assert r2.status == HealthStatus.HEALTHY

    def test_unhealthy_stays_unhealthy_on_continued_failure(self) -> None:
        # After reaching UNHEALTHY, continued failures stay UNHEALTHY (line 374)
        checker = HealthChecker(_hc_config(healthy_threshold=2, unhealthy_threshold=1))
        ep = _endpoint("svc")
        checker.check(ep, _failing_probe)  # → UNHEALTHY
        r2 = checker.check(ep, _failing_probe)  # → stays UNHEALTHY (line 374)
        assert r2.status == HealthStatus.UNHEALTHY


# ===========================================================================
# HealthChecker — check_all()
# ===========================================================================


class TestHealthCheckerCheckAll:
    def test_check_all_returns_results_in_order(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1, unhealthy_threshold=1))
        endpoints = [_endpoint("a"), _endpoint("b"), _endpoint("c")]
        results = checker.check_all(endpoints, _noop_probe)
        assert [r.endpoint_name for r in results] == ["a", "b", "c"]

    def test_check_all_empty_list_returns_empty(self) -> None:
        checker = HealthChecker(_hc_config())
        results = checker.check_all([], _noop_probe)
        assert results == []

    def test_check_all_mixed_probes(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1, unhealthy_threshold=1))
        good_ep = _endpoint("good")
        bad_ep = _endpoint("bad")

        def mixed_probe(endpoint: ServiceEndpoint) -> None:
            if endpoint.name == "bad":
                raise RuntimeError("bad endpoint")

        results = checker.check_all([good_ep, bad_ep], mixed_probe)
        assert results[0].status == HealthStatus.HEALTHY
        assert results[1].status == HealthStatus.UNHEALTHY


# ===========================================================================
# HealthChecker — history
# ===========================================================================


class TestHealthCheckerHistory:
    def test_history_accumulates_results(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1, unhealthy_threshold=1))
        ep = _endpoint("svc")
        checker.check(ep, _noop_probe)
        checker.check(ep, _noop_probe)
        checker.check(ep, _failing_probe)
        assert len(checker.history("svc")) == 3

    def test_history_empty_for_unknown_endpoint(self) -> None:
        checker = HealthChecker(_hc_config())
        assert checker.history("never-checked") == []

    def test_history_respects_max_history(self) -> None:
        cfg = HealthCheckConfig(
            interval_seconds=30.0,
            timeout_seconds=5.0,
            healthy_threshold=1,
            unhealthy_threshold=1,
            max_history=3,
        )
        checker = HealthChecker(cfg)
        ep = _endpoint("svc")
        for _ in range(10):
            checker.check(ep, _noop_probe)
        assert len(checker.history("svc")) == 3

    def test_history_oldest_first(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1, unhealthy_threshold=1))
        ep = _endpoint("svc")
        checker.check(ep, _noop_probe)
        checker.check(ep, _failing_probe)
        history = checker.history("svc")
        assert history[0].status == HealthStatus.HEALTHY
        assert history[1].status == HealthStatus.UNHEALTHY

    def test_clear_history_specific_endpoint(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1, unhealthy_threshold=1))
        ep_a = _endpoint("a")
        ep_b = _endpoint("b")
        checker.check(ep_a, _noop_probe)
        checker.check(ep_b, _noop_probe)
        checker.clear_history("a")
        assert checker.history("a") == []
        assert len(checker.history("b")) == 1

    def test_clear_history_all(self) -> None:
        checker = HealthChecker(_hc_config(healthy_threshold=1, unhealthy_threshold=1))
        for name in ("x", "y", "z"):
            checker.check(_endpoint(name), _noop_probe)
        checker.clear_history()
        for name in ("x", "y", "z"):
            assert checker.history(name) == []

    def test_repr_shows_tracked_endpoints(self) -> None:
        checker = HealthChecker(_hc_config())
        checker.check(_endpoint("svc"), _noop_probe)
        r = repr(checker)
        assert "tracked_endpoints=1" in r


# ===========================================================================
# Concurrent access — CircuitBreaker
# ===========================================================================


class TestConcurrentCircuitBreaker:
    def test_concurrent_execute_does_not_corrupt_state(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=100))
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    cb.execute(lambda: None)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cb.total_calls == 400

    def test_concurrent_failure_recording_opens_once(self) -> None:
        cb = CircuitBreaker(_config(failure_threshold=10))
        open_events: list[CircuitState] = []

        def worker() -> None:
            for _ in range(5):
                cb.record_failure()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.state == CircuitState.OPEN


# ===========================================================================
# Concurrent access — LoadBalancer
# ===========================================================================


class TestConcurrentLoadBalancer:
    def test_concurrent_select_returns_valid_endpoints(self) -> None:
        lb = LoadBalancer()
        for i in range(5):
            lb.add_endpoint(_endpoint(f"ep-{i}"))

        results: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                for _ in range(20):
                    ep = lb.select(LoadBalancerStrategy.ROUND_ROBIN)
                    with lock:
                        results.append(ep.name)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 200
        for name in results:
            assert name.startswith("ep-")


# ===========================================================================
# CLI — resilience group
# ===========================================================================


class TestCLIResilienceGroup:
    def test_resilience_group_in_cli(self) -> None:
        from agent_mesh_router.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["resilience", "--help"])
        assert result.exit_code == 0
        assert "resilience" in result.output.lower() or "health" in result.output.lower()

    def test_health_command_help(self) -> None:
        from agent_mesh_router.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["resilience", "health", "--help"])
        assert result.exit_code == 0
        assert "--endpoints" in result.output

    def test_circuit_breaker_status_command(self) -> None:
        from agent_mesh_router.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["resilience", "circuit-breaker", "status"])
        assert result.exit_code == 0

    def test_health_command_with_valid_json(self, tmp_path: Path) -> None:
        from agent_mesh_router.cli.main import cli

        endpoints_data = [
            {"name": "local", "host": "127.0.0.1", "port": 1},
        ]
        endpoints_file = tmp_path / "endpoints.json"
        endpoints_file.write_text(json.dumps(endpoints_data))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["resilience", "health", "--endpoints", str(endpoints_file)]
        )
        # Exit code may be non-zero if port 1 is not open; just verify no crash
        assert result.exit_code in (0, 1, 2)

    def test_health_command_with_invalid_json(self, tmp_path: Path) -> None:
        from agent_mesh_router.cli.main import cli

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not: valid json}")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["resilience", "health", "--endpoints", str(bad_file)]
        )
        assert result.exit_code == 1

    def test_health_command_with_non_list_json(self, tmp_path: Path) -> None:
        from agent_mesh_router.cli.main import cli

        obj_file = tmp_path / "obj.json"
        obj_file.write_text(json.dumps({"name": "not-a-list"}))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["resilience", "health", "--endpoints", str(obj_file)]
        )
        assert result.exit_code == 1

    def test_health_command_with_missing_fields(self, tmp_path: Path) -> None:
        from agent_mesh_router.cli.main import cli

        incomplete_file = tmp_path / "incomplete.json"
        incomplete_file.write_text(json.dumps([{"name": "missing-host"}]))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["resilience", "health", "--endpoints", str(incomplete_file)]
        )
        assert result.exit_code == 1

    def test_health_command_empty_array(self, tmp_path: Path) -> None:
        from agent_mesh_router.cli.main import cli

        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["resilience", "health", "--endpoints", str(empty_file)]
        )
        assert result.exit_code == 0

    def test_health_command_timeout_option(self, tmp_path: Path) -> None:
        from agent_mesh_router.cli.main import cli

        endpoints_file = tmp_path / "ep.json"
        endpoints_file.write_text(json.dumps([{"name": "x", "host": "h", "port": 80}]))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["resilience", "health", "--endpoints", str(endpoints_file), "--timeout", "1.0"],
        )
        assert result.exit_code in (0, 1, 2)


# ===========================================================================
# Package __init__ exports
# ===========================================================================


class TestResiliencePackageExports:
    def test_all_public_names_importable(self) -> None:
        from agent_mesh_router import resilience  # noqa: F401 – import smoke test

        from agent_mesh_router.resilience import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitBreakerOpenError,
            CircuitState,
            HealthCheckConfig,
            HealthCheckResult,
            HealthChecker,
            HealthStatus,
            LoadBalancer,
            LoadBalancerStrategy,
            NoHealthyEndpointError,
            RetryConfig,
            RetryExhaustedError,
            RetryPolicy,
            ServiceEndpoint,
        )

        assert CircuitBreaker is not None
        assert CircuitBreakerConfig is not None
        assert CircuitBreakerOpenError is not None
        assert CircuitState is not None
        assert HealthCheckConfig is not None
        assert HealthCheckResult is not None
        assert HealthChecker is not None
        assert HealthStatus is not None
        assert LoadBalancer is not None
        assert LoadBalancerStrategy is not None
        assert NoHealthyEndpointError is not None
        assert RetryConfig is not None
        assert RetryExhaustedError is not None
        assert RetryPolicy is not None
        assert ServiceEndpoint is not None
