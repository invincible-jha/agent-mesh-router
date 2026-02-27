"""Backpressure management — queue monitoring and adaptive rate limiting.

This package provides two components:

monitor
    ``BackpressureMonitor`` — tracks named queue depths and computes a
    normalized ``pressure_score`` in [0.0, 1.0].  Thresholds at 0.7 (warn)
    and 0.9 (critical) are enforced.

throttle
    ``AdaptiveThrottle`` — per-agent token-bucket rate limiter that
    automatically reduces its refill rate when the system is under pressure.

Example
-------
::

    from agent_mesh_router.backpressure import (
        BackpressureMonitor,
        AdaptiveThrottle,
        WARN_THRESHOLD,
        CRITICAL_THRESHOLD,
    )

    monitor = BackpressureMonitor()
    monitor.register_queue("main", max_depth=1000)
    monitor.update_depth("main", 800)

    throttle = AdaptiveThrottle("agent-a", base_rate=100.0, monitor=monitor)
    if not throttle.try_acquire():
        print("Throttled due to pressure:", monitor.pressure_score())
"""
from __future__ import annotations

from agent_mesh_router.backpressure.monitor import (
    CRITICAL_THRESHOLD,
    WARN_THRESHOLD,
    BackpressureMonitor,
    QueueMetrics,
)
from agent_mesh_router.backpressure.throttle import AdaptiveThrottle
from agent_mesh_router.backpressure.task_queue import (
    BackpressureQueue,
    QueueFullError,
    QueueEmptyError,
    QueueStats,
)

__all__ = [
    "AdaptiveThrottle",
    "BackpressureMonitor",
    "BackpressureQueue",
    "CRITICAL_THRESHOLD",
    "QueueEmptyError",
    "QueueFullError",
    "QueueMetrics",
    "QueueStats",
    "WARN_THRESHOLD",
]
