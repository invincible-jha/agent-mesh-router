"""Tests for recovery_tester module.

Covers:
- RecoveryTestResult frozen dataclass validation
- RecoveryTestResult.success_rate property
- RecoveryTester construction validation
- RecoveryTester.test_recovery with all-success scenario
- RecoveryTester.test_recovery with all-failure scenario
- RecoveryTester.test_recovery with partial success
- Early stop on first failure (stop_on_first_failure=True)
- Continued probing on failure (stop_on_first_failure=False)
- repr() outputs
- Integration with circuit breaker state logic
"""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from agent_mesh_router.resilience.recovery_tester import (
    RecoveryTestResult,
    RecoveryTester,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _always_succeed() -> str:
    return "ok"


def _always_fail() -> None:
    raise RuntimeError("downstream unavailable")


def _fail_then_succeed(num_failures: int) -> callable:
    """Return a function that fails for the first `num_failures` calls then succeeds."""
    counter = {"count": 0}

    def fn() -> str:
        counter["count"] += 1
        if counter["count"] <= num_failures:
            raise RuntimeError(f"fail #{counter['count']}")
        return "recovered"

    return fn


# ---------------------------------------------------------------------------
# RecoveryTestResult
# ---------------------------------------------------------------------------


class TestRecoveryTestResult:
    def test_valid_construction(self) -> None:
        result = RecoveryTestResult(
            total_attempts=3,
            successes=2,
            failures=1,
            should_close=True,
            latency_ms=15.0,
        )
        assert result.total_attempts == 3
        assert result.successes == 2
        assert result.failures == 1
        assert result.should_close is True
        assert result.latency_ms == 15.0

    def test_is_frozen(self) -> None:
        result = RecoveryTestResult(
            total_attempts=1, successes=1, failures=0,
            should_close=True, latency_ms=1.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.successes = 99  # type: ignore[misc]

    def test_negative_total_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="total_attempts"):
            RecoveryTestResult(
                total_attempts=-1, successes=0, failures=0,
                should_close=False, latency_ms=0.0,
            )

    def test_negative_successes_raises(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            RecoveryTestResult(
                total_attempts=1, successes=-1, failures=0,
                should_close=False, latency_ms=0.0,
            )

    def test_negative_failures_raises(self) -> None:
        with pytest.raises(ValueError, match="failures"):
            RecoveryTestResult(
                total_attempts=1, successes=0, failures=-1,
                should_close=False, latency_ms=0.0,
            )

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency_ms"):
            RecoveryTestResult(
                total_attempts=1, successes=1, failures=0,
                should_close=True, latency_ms=-0.1,
            )

    def test_successes_plus_failures_exceeds_total_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds total_attempts"):
            RecoveryTestResult(
                total_attempts=2, successes=2, failures=1,
                should_close=True, latency_ms=0.0,
            )

    def test_success_rate_all_success(self) -> None:
        result = RecoveryTestResult(
            total_attempts=4, successes=4, failures=0,
            should_close=True, latency_ms=0.0,
        )
        assert result.success_rate == 1.0

    def test_success_rate_all_failure(self) -> None:
        result = RecoveryTestResult(
            total_attempts=3, successes=0, failures=3,
            should_close=False, latency_ms=0.0,
        )
        assert result.success_rate == 0.0

    def test_success_rate_partial(self) -> None:
        result = RecoveryTestResult(
            total_attempts=4, successes=2, failures=2,
            should_close=False, latency_ms=0.0,
        )
        assert result.success_rate == pytest.approx(0.5)

    def test_success_rate_zero_attempts(self) -> None:
        result = RecoveryTestResult(
            total_attempts=0, successes=0, failures=0,
            should_close=False, latency_ms=0.0,
        )
        assert result.success_rate == 0.0


# ---------------------------------------------------------------------------
# RecoveryTester construction
# ---------------------------------------------------------------------------


class TestRecoveryTesterConstruction:
    def test_default_parameters(self) -> None:
        tester = RecoveryTester()
        assert tester.max_test_requests == 3
        assert tester.success_threshold == 2

    def test_custom_parameters(self) -> None:
        tester = RecoveryTester(max_test_requests=5, success_threshold=3)
        assert tester.max_test_requests == 5
        assert tester.success_threshold == 3

    def test_max_test_requests_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_test_requests"):
            RecoveryTester(max_test_requests=0)

    def test_success_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="success_threshold"):
            RecoveryTester(success_threshold=0)

    def test_success_threshold_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="success_threshold"):
            RecoveryTester(max_test_requests=2, success_threshold=3)

    def test_threshold_equals_max_is_valid(self) -> None:
        tester = RecoveryTester(max_test_requests=3, success_threshold=3)
        assert tester.success_threshold == 3

    def test_repr_contains_config(self) -> None:
        tester = RecoveryTester(max_test_requests=5, success_threshold=3)
        representation = repr(tester)
        assert "5" in representation
        assert "3" in representation


# ---------------------------------------------------------------------------
# RecoveryTester.test_recovery
# ---------------------------------------------------------------------------


class TestRecoveryTesterProbing:
    def test_all_successes_should_close_true(self) -> None:
        tester = RecoveryTester(max_test_requests=3, success_threshold=2)
        result = tester.test_recovery(_always_succeed)
        assert result.should_close is True
        assert result.successes >= 2

    def test_all_failures_should_close_false(self) -> None:
        tester = RecoveryTester(max_test_requests=3, success_threshold=2)
        result = tester.test_recovery(_always_fail)
        assert result.should_close is False
        assert result.failures >= 1

    def test_stop_on_first_failure_limits_attempts(self) -> None:
        tester = RecoveryTester(
            max_test_requests=5,
            success_threshold=3,
            stop_on_first_failure=True,
        )
        call_count = {"n": 0}

        def counted_fail() -> None:
            call_count["n"] += 1
            raise RuntimeError("fail")

        result = tester.test_recovery(counted_fail)
        # With stop_on_first_failure, only 1 call should occur
        assert call_count["n"] == 1
        assert result.total_attempts == 1
        assert result.failures == 1

    def test_no_stop_on_first_failure_continues_probing(self) -> None:
        tester = RecoveryTester(
            max_test_requests=4,
            success_threshold=4,  # threshold equals max so all 4 must run
            stop_on_first_failure=False,
        )
        call_count = {"n": 0}

        def counted_fail() -> None:
            call_count["n"] += 1
            raise RuntimeError("fail")

        result = tester.test_recovery(counted_fail)
        assert call_count["n"] == 4
        assert result.total_attempts == 4

    def test_stops_early_when_threshold_met(self) -> None:
        tester = RecoveryTester(max_test_requests=5, success_threshold=2)
        call_count = {"n": 0}

        def counted_succeed() -> str:
            call_count["n"] += 1
            return "ok"

        result = tester.test_recovery(counted_succeed)
        # Should stop after 2 successes
        assert call_count["n"] == 2
        assert result.successes == 2
        assert result.should_close is True

    def test_latency_ms_is_non_negative(self) -> None:
        tester = RecoveryTester()
        result = tester.test_recovery(_always_succeed)
        assert result.latency_ms >= 0.0

    def test_args_and_kwargs_forwarded(self) -> None:
        received: list[tuple] = []

        def capturing_fn(pos_arg: str, *, kw_arg: int) -> str:
            received.append((pos_arg, kw_arg))
            return "ok"

        tester = RecoveryTester(max_test_requests=1, success_threshold=1)
        tester.test_recovery(capturing_fn, "hello", kw_arg=42)
        assert received == [("hello", 42)]

    def test_partial_success_below_threshold(self) -> None:
        # Fails on first attempt, then succeeds — but threshold=3 not met
        tester = RecoveryTester(
            max_test_requests=3,
            success_threshold=3,
            stop_on_first_failure=False,
        )
        fn = _fail_then_succeed(1)
        result = tester.test_recovery(fn)
        # 1 failure + 2 successes = 3 total; threshold=3 not met
        assert result.failures == 1
        assert result.successes == 2
        assert result.should_close is False

    def test_result_total_attempts_equals_successes_plus_failures(self) -> None:
        tester = RecoveryTester(
            max_test_requests=4,
            success_threshold=4,
            stop_on_first_failure=False,
        )
        fn = _fail_then_succeed(2)
        result = tester.test_recovery(fn)
        assert result.total_attempts == result.successes + result.failures

    def test_mock_service_called_correct_times(self) -> None:
        mock_fn = MagicMock(return_value="data")
        tester = RecoveryTester(max_test_requests=3, success_threshold=2)
        result = tester.test_recovery(mock_fn)
        # Should stop after 2 successes
        assert mock_fn.call_count == 2
        assert result.should_close is True
