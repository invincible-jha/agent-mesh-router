"""Tests for agent_mesh_router.backpressure.task_queue — BackpressureQueue."""
from __future__ import annotations

import threading
import time

import pytest

from agent_mesh_router.backpressure.monitor import BackpressureMonitor
from agent_mesh_router.backpressure.task_queue import (
    BackpressureQueue,
    QueueEmptyError,
    QueueFullError,
    QueueStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> None:
    pass


def _make_queue(max_size: int = 10, **kwargs: object) -> BackpressureQueue:
    return BackpressureQueue(max_size=max_size, **kwargs)  # type: ignore[arg-type]


# ===========================================================================
# QueueStats
# ===========================================================================


class TestQueueStats:
    def test_frozen(self) -> None:
        stats = QueueStats(
            size=5,
            max_size=10,
            fill_ratio=0.5,
            is_under_pressure=False,
            enqueued_total=5,
            dequeued_total=0,
            dropped_total=0,
        )
        with pytest.raises(Exception):
            stats.size = 99  # type: ignore[misc]


# ===========================================================================
# Construction
# ===========================================================================


class TestBackpressureQueueConstruction:
    def test_default_construction(self) -> None:
        bq = BackpressureQueue()
        assert bq.max_size == 100
        assert bq.size == 0

    def test_custom_max_size(self) -> None:
        bq = BackpressureQueue(max_size=5)
        assert bq.max_size == 5

    def test_zero_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            BackpressureQueue(max_size=0)

    def test_negative_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            BackpressureQueue(max_size=-1)

    def test_invalid_high_watermark_raises(self) -> None:
        with pytest.raises(ValueError, match="high_watermark"):
            BackpressureQueue(high_watermark=0.0)

    def test_invalid_low_watermark_raises(self) -> None:
        with pytest.raises(ValueError, match="low_watermark"):
            BackpressureQueue(high_watermark=0.8, low_watermark=0.9)

    def test_monitor_registered(self) -> None:
        monitor = BackpressureMonitor()
        bq = BackpressureQueue(max_size=10, monitor=monitor, queue_name="test_q")
        metrics = monitor.queue_metrics("test_q")
        assert metrics.max_depth == 10

    def test_default_monitor_created(self) -> None:
        bq = BackpressureQueue(max_size=10)
        assert bq.monitor is not None


# ===========================================================================
# Enqueue
# ===========================================================================


class TestEnqueue:
    def test_enqueue_increases_size(self) -> None:
        bq = _make_queue(5)
        bq.enqueue(_noop)
        assert bq.size == 1

    def test_enqueue_multiple(self) -> None:
        bq = _make_queue(5)
        for _ in range(3):
            bq.enqueue(_noop)
        assert bq.size == 3

    def test_enqueue_non_callable_raises(self) -> None:
        bq = _make_queue(5)
        with pytest.raises(TypeError):
            bq.enqueue("not_a_task")  # type: ignore[arg-type]

    def test_enqueue_full_nonblocking_raises(self) -> None:
        bq = _make_queue(2)
        bq.enqueue(_noop)
        bq.enqueue(_noop)
        with pytest.raises(QueueFullError):
            bq.enqueue_nowait(_noop)

    def test_enqueue_full_increments_dropped(self) -> None:
        bq = _make_queue(1)
        bq.enqueue(_noop)
        with pytest.raises(QueueFullError):
            bq.enqueue_nowait(_noop)
        stats = bq.stats()
        assert stats.dropped_total == 1

    def test_enqueue_updates_monitor(self) -> None:
        monitor = BackpressureMonitor()
        bq = BackpressureQueue(max_size=10, monitor=monitor, queue_name="m_q")
        bq.enqueue(_noop)
        metrics = monitor.queue_metrics("m_q")
        assert metrics.current_depth == 1


# ===========================================================================
# Dequeue
# ===========================================================================


class TestDequeue:
    def test_dequeue_returns_task(self) -> None:
        bq = _make_queue(5)
        results: list[int] = []
        bq.enqueue(lambda: results.append(1))
        task = bq.dequeue()
        task()
        assert results == [1]

    def test_dequeue_fifo_order(self) -> None:
        bq = _make_queue(5)
        order: list[int] = []
        bq.enqueue(lambda: order.append(1))
        bq.enqueue(lambda: order.append(2))
        bq.enqueue(lambda: order.append(3))
        for _ in range(3):
            bq.dequeue()()
        assert order == [1, 2, 3]

    def test_dequeue_empty_nonblocking_raises(self) -> None:
        bq = _make_queue(5)
        with pytest.raises(QueueEmptyError):
            bq.dequeue_nowait()

    def test_dequeue_decreases_size(self) -> None:
        bq = _make_queue(5)
        bq.enqueue(_noop)
        bq.enqueue(_noop)
        bq.dequeue()
        assert bq.size == 1

    def test_dequeue_updates_monitor(self) -> None:
        monitor = BackpressureMonitor()
        bq = BackpressureQueue(max_size=10, monitor=monitor, queue_name="m_q2")
        bq.enqueue(_noop)
        bq.dequeue()
        metrics = monitor.queue_metrics("m_q2")
        assert metrics.current_depth == 0


# ===========================================================================
# Pressure watermarks
# ===========================================================================


class TestPressureWatermarks:
    def test_not_under_pressure_initially(self) -> None:
        bq = BackpressureQueue(max_size=10, high_watermark=0.8)
        assert not bq.is_under_pressure

    def test_under_pressure_when_high_watermark_reached(self) -> None:
        bq = BackpressureQueue(max_size=10, high_watermark=0.8, low_watermark=0.5)
        for _ in range(8):  # 8/10 = 0.8 — at threshold
            bq.enqueue(_noop)
        assert bq.is_under_pressure

    def test_pressure_relieved_below_low_watermark(self) -> None:
        bq = BackpressureQueue(max_size=10, high_watermark=0.8, low_watermark=0.5)
        for _ in range(8):
            bq.enqueue(_noop)
        assert bq.is_under_pressure
        # Drain below low watermark (5/10 = 0.5)
        for _ in range(4):
            bq.dequeue()
        assert not bq.is_under_pressure


# ===========================================================================
# Stats
# ===========================================================================


class TestStats:
    def test_stats_initial(self) -> None:
        bq = _make_queue(10)
        stats = bq.stats()
        assert stats.size == 0
        assert stats.max_size == 10
        assert stats.fill_ratio == 0.0
        assert stats.enqueued_total == 0
        assert stats.dequeued_total == 0
        assert stats.dropped_total == 0

    def test_stats_after_enqueue(self) -> None:
        bq = _make_queue(10)
        bq.enqueue(_noop)
        bq.enqueue(_noop)
        stats = bq.stats()
        assert stats.enqueued_total == 2
        assert stats.size == 2

    def test_stats_after_dequeue(self) -> None:
        bq = _make_queue(10)
        bq.enqueue(_noop)
        bq.dequeue()
        stats = bq.stats()
        assert stats.dequeued_total == 1
        assert stats.size == 0

    def test_fill_ratio(self) -> None:
        bq = _make_queue(10)
        for _ in range(5):
            bq.enqueue(_noop)
        assert abs(bq.fill_ratio - 0.5) < 1e-9

    def test_returns_queue_stats_type(self) -> None:
        bq = _make_queue(10)
        assert isinstance(bq.stats(), QueueStats)


# ===========================================================================
# Drain
# ===========================================================================


class TestDrain:
    def test_drain_returns_all_tasks(self) -> None:
        bq = _make_queue(10)
        counter: list[int] = []
        for i in range(5):
            i_copy = i
            bq.enqueue(lambda x=i_copy: counter.append(x))
        tasks = bq.drain()
        assert len(tasks) == 5
        for t in tasks:
            t()
        assert sorted(counter) == [0, 1, 2, 3, 4]

    def test_drain_empties_queue(self) -> None:
        bq = _make_queue(10)
        bq.enqueue(_noop)
        bq.drain()
        assert bq.size == 0
        assert bq.is_empty

    def test_drain_empty_queue_returns_empty_list(self) -> None:
        bq = _make_queue(10)
        result = bq.drain()
        assert result == []


# ===========================================================================
# Properties and dunder
# ===========================================================================


class TestProperties:
    def test_is_empty_initial(self) -> None:
        bq = _make_queue(5)
        assert bq.is_empty

    def test_is_full(self) -> None:
        bq = _make_queue(2)
        bq.enqueue(_noop)
        bq.enqueue(_noop)
        assert bq.is_full

    def test_len(self) -> None:
        bq = _make_queue(5)
        bq.enqueue(_noop)
        bq.enqueue(_noop)
        assert len(bq) == 2

    def test_repr(self) -> None:
        bq = BackpressureQueue(max_size=10, queue_name="repr_test")
        r = repr(bq)
        assert "repr_test" in r
        assert "10" in r


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_enqueue_dequeue(self) -> None:
        bq = _make_queue(1000)
        errors: list[Exception] = []
        enqueued: list[None] = []
        dequeued: list[None] = []

        def producer() -> None:
            try:
                for _ in range(50):
                    bq.enqueue(lambda: None)
                    enqueued.append(None)
            except Exception as exc:
                errors.append(exc)

        def consumer() -> None:
            try:
                for _ in range(50):
                    bq.dequeue(timeout=2.0)
                    dequeued.append(None)
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=producer) for _ in range(5)]
            + [threading.Thread(target=consumer) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        stats = bq.stats()
        assert stats.enqueued_total == 250
        assert stats.dequeued_total == 250
