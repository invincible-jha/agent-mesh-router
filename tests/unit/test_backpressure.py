"""Tests for backpressure monitor and adaptive throttle."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from agent_mesh_router.backpressure.monitor import (
    CRITICAL_THRESHOLD,
    WARN_THRESHOLD,
    BackpressureMonitor,
    QueueMetrics,
)
from agent_mesh_router.backpressure.throttle import AdaptiveThrottle


# ---------------------------------------------------------------------------
# QueueMetrics
# ---------------------------------------------------------------------------


class TestQueueMetrics:
    def test_utilization_unbounded(self) -> None:
        m = QueueMetrics(name="q", max_depth=0)
        assert m.utilization == 0.0

    def test_utilization_empty(self) -> None:
        m = QueueMetrics(name="q", max_depth=100, current_depth=0)
        assert m.utilization == 0.0

    def test_utilization_partial(self) -> None:
        m = QueueMetrics(name="q", max_depth=100, current_depth=50)
        assert m.utilization == pytest.approx(0.5)

    def test_utilization_full(self) -> None:
        m = QueueMetrics(name="q", max_depth=100, current_depth=100)
        assert m.utilization == 1.0

    def test_utilization_capped_at_one(self) -> None:
        m = QueueMetrics(name="q", max_depth=100, current_depth=200)
        assert m.utilization == 1.0


# ---------------------------------------------------------------------------
# BackpressureMonitor
# ---------------------------------------------------------------------------


class TestBackpressureMonitor:
    def test_default_construction(self) -> None:
        monitor = BackpressureMonitor()
        assert monitor.warn_threshold == WARN_THRESHOLD
        assert monitor.critical_threshold == CRITICAL_THRESHOLD

    def test_invalid_threshold_order(self) -> None:
        with pytest.raises(ValueError):
            BackpressureMonitor(warn_threshold=0.9, critical_threshold=0.5)

    def test_pressure_score_no_queues(self) -> None:
        monitor = BackpressureMonitor()
        assert monitor.pressure_score() == 0.0

    def test_register_and_update(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q1", max_depth=100)
        monitor.update_depth("q1", 75)
        assert monitor.pressure_score() == pytest.approx(0.75)

    def test_is_warn_true(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 80)
        assert monitor.is_warn() is True

    def test_is_warn_false(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 50)
        assert monitor.is_warn() is False

    def test_is_critical_true(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 95)
        assert monitor.is_critical() is True

    def test_is_critical_false(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 80)
        assert monitor.is_critical() is False

    def test_deregister_queue(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 100)
        monitor.deregister_queue("q")
        assert monitor.pressure_score() == 0.0

    def test_deregister_nonexistent_silently(self) -> None:
        monitor = BackpressureMonitor()
        monitor.deregister_queue("nonexistent")  # Should not raise.

    def test_update_depth_negative_raises(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        with pytest.raises(ValueError):
            monitor.update_depth("q", -1)

    def test_update_depth_unknown_raises(self) -> None:
        monitor = BackpressureMonitor()
        with pytest.raises(KeyError):
            monitor.update_depth("nonexistent", 10)

    def test_update_depths_bulk(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q1", max_depth=100)
        monitor.register_queue("q2", max_depth=100)
        monitor.update_depths({"q1": 50, "q2": 100, "unknown": 999})
        q1 = monitor.queue_metrics("q1")
        q2 = monitor.queue_metrics("q2")
        assert q1.current_depth == 50
        assert q2.current_depth == 100

    def test_queue_metrics_unknown_raises(self) -> None:
        monitor = BackpressureMonitor()
        with pytest.raises(KeyError):
            monitor.queue_metrics("nope")

    def test_all_metrics(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q1", max_depth=100)
        monitor.register_queue("q2", max_depth=200)
        metrics = monitor.all_metrics()
        assert len(metrics) == 2

    def test_weighted_pressure(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q1", max_depth=100, weight=1.0)
        monitor.register_queue("q2", max_depth=100, weight=3.0)
        monitor.update_depth("q1", 100)
        monitor.update_depth("q2", 0)
        # Weighted avg = (1.0 * 1.0 + 0.0 * 3.0) / (1 + 3) = 0.25
        assert monitor.pressure_score() == pytest.approx(0.25)

    def test_zero_weight_all_queues_returns_zero(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q1", max_depth=100, weight=0.0)
        monitor.update_depth("q1", 100)
        assert monitor.pressure_score() == 0.0

    def test_register_invalid_weight(self) -> None:
        monitor = BackpressureMonitor()
        with pytest.raises(ValueError):
            monitor.register_queue("q", weight=-1.0)

    def test_register_invalid_max_depth(self) -> None:
        monitor = BackpressureMonitor()
        with pytest.raises(ValueError):
            monitor.register_queue("q", max_depth=-1)

    def test_repr(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        repr_str = repr(monitor)
        assert "BackpressureMonitor" in repr_str
        assert "queues=1" in repr_str

    def test_reregister_overwrites(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 100)
        monitor.register_queue("q", max_depth=200)
        # After re-register, current_depth resets to 0.
        assert monitor.pressure_score() == 0.0


# ---------------------------------------------------------------------------
# AdaptiveThrottle
# ---------------------------------------------------------------------------


class TestAdaptiveThrottle:
    def test_default_construction(self) -> None:
        t = AdaptiveThrottle("agent-x", base_rate=100.0)
        assert t.agent_id == "agent-x"
        assert t.base_rate == 100.0

    def test_invalid_base_rate(self) -> None:
        with pytest.raises(ValueError):
            AdaptiveThrottle("a", base_rate=0.0)

    def test_invalid_bucket_capacity(self) -> None:
        with pytest.raises(ValueError):
            AdaptiveThrottle("a", base_rate=10.0, bucket_capacity=0)

    def test_invalid_min_rate_fraction(self) -> None:
        with pytest.raises(ValueError):
            AdaptiveThrottle("a", base_rate=10.0, min_rate_fraction=0.0)

    def test_try_acquire_success_from_full_bucket(self) -> None:
        t = AdaptiveThrottle("a", base_rate=100.0, bucket_capacity=200)
        assert t.try_acquire() is True
        assert t.total_acquired == 1
        assert t.total_rejected == 0

    def test_try_acquire_rejected_when_empty(self) -> None:
        t = AdaptiveThrottle("a", base_rate=100.0, bucket_capacity=1)
        t.try_acquire()  # Uses last token.
        result = t.try_acquire(tokens=1.0)
        # May succeed if refill adds tokens; use large demand to force failure.
        t2 = AdaptiveThrottle("a", base_rate=0.001, bucket_capacity=1)
        t2.try_acquire()  # Drain.
        assert t2.try_acquire(tokens=1.0) is False
        assert t2.total_rejected >= 1

    def test_available_tokens_non_negative(self) -> None:
        t = AdaptiveThrottle("a", base_rate=100.0, bucket_capacity=50)
        tokens = t.available_tokens()
        assert 0.0 <= tokens <= 50.0

    def test_effective_rate_no_monitor(self) -> None:
        t = AdaptiveThrottle("a", base_rate=100.0)
        assert t.effective_rate() == pytest.approx(100.0)

    def test_effective_rate_below_warn(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 10)
        t = AdaptiveThrottle("a", base_rate=100.0, monitor=monitor)
        # pressure < 0.7, so full rate.
        assert t.effective_rate() == pytest.approx(100.0)

    def test_effective_rate_at_critical(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 95)
        t = AdaptiveThrottle(
            "a", base_rate=100.0, monitor=monitor, min_rate_fraction=0.1
        )
        assert t.effective_rate() == pytest.approx(10.0)

    def test_effective_rate_in_warn_band(self) -> None:
        monitor = BackpressureMonitor()
        monitor.register_queue("q", max_depth=100)
        monitor.update_depth("q", 80)  # pressure = 0.8, midway in warn band
        t = AdaptiveThrottle(
            "a",
            base_rate=100.0,
            monitor=monitor,
            min_rate_fraction=0.1,
            warn_threshold=0.7,
            critical_threshold=0.9,
        )
        rate = t.effective_rate()
        assert 10.0 < rate < 100.0

    def test_repr(self) -> None:
        t = AdaptiveThrottle("agent-z", base_rate=50.0)
        assert "agent-z" in repr(t)

    @pytest.mark.asyncio
    async def test_acquire_async_succeeds_immediately(self) -> None:
        t = AdaptiveThrottle("a", base_rate=100.0, bucket_capacity=100)
        result = await t.acquire_async(tokens=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_async_timeout(self) -> None:
        t = AdaptiveThrottle("a", base_rate=0.001, bucket_capacity=1)
        t.try_acquire()  # Drain bucket.
        result = await t.acquire_async(tokens=1.0, timeout=0.05)
        assert result is False

    def test_properties(self) -> None:
        t = AdaptiveThrottle("x", base_rate=10.0, bucket_capacity=20)
        assert t.bucket_capacity == pytest.approx(20.0)
        assert t.total_acquired == 0
        assert t.total_rejected == 0
