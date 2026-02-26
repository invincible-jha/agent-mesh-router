"""BackpressureMonitor — track queue depths and compute pressure scores.

The monitor watches registered queues and computes a normalized
``pressure_score`` in [0.0, 1.0] that aggregates utilization across all
monitored queues.  Callers can interrogate the score directly or register
callback listeners that fire when thresholds are crossed.

Thresholds
----------
``WARN_THRESHOLD = 0.7``
    Pressure above this level indicates the system is approaching its limit.
``CRITICAL_THRESHOLD = 0.9``
    Pressure above this level indicates the system is overloaded.

Pressure calculation
--------------------
For each queue the utilization is ``current_depth / max_depth``.  When
``max_depth`` is 0 (unbounded), utilization is reported as 0.0.
The overall ``pressure_score`` is the weighted average utilization across
all queues (equal weights by default).

Example
-------
::

    from agent_mesh_router.backpressure.monitor import BackpressureMonitor

    monitor = BackpressureMonitor()
    monitor.register_queue("broker-main", max_depth=1000)
    monitor.update_depth("broker-main", 750)

    print(monitor.pressure_score())  # 0.75
    print(monitor.is_critical())     # False
    print(monitor.is_warn())         # True
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WARN_THRESHOLD: float = 0.7
CRITICAL_THRESHOLD: float = 0.9


@dataclass
class QueueMetrics:
    """Snapshot of a single monitored queue's utilization.

    Attributes
    ----------
    name:
        Logical name of the queue.
    current_depth:
        Current number of items in the queue.
    max_depth:
        Maximum queue capacity.  0 indicates unbounded.
    utilization:
        Fraction full (0.0–1.0).  Always 0.0 for unbounded queues.
    weight:
        Contribution weight for the aggregate pressure score.
    """

    name: str
    current_depth: int = 0
    max_depth: int = 0
    weight: float = 1.0

    @property
    def utilization(self) -> float:
        """Return queue fill fraction in [0.0, 1.0]."""
        if self.max_depth <= 0:
            return 0.0
        return min(1.0, self.current_depth / self.max_depth)


class BackpressureMonitor:
    """Track queue depths across multiple named queues and compute pressure.

    All methods are thread-safe.

    Parameters
    ----------
    warn_threshold:
        Pressure score above which ``is_warn()`` returns True.
        Defaults to ``WARN_THRESHOLD`` (0.7).
    critical_threshold:
        Pressure score above which ``is_critical()`` returns True.
        Defaults to ``CRITICAL_THRESHOLD`` (0.9).
    """

    def __init__(
        self,
        *,
        warn_threshold: float = WARN_THRESHOLD,
        critical_threshold: float = CRITICAL_THRESHOLD,
    ) -> None:
        if not (0.0 <= warn_threshold < critical_threshold <= 1.0):
            raise ValueError(
                f"Must have 0 <= warn_threshold < critical_threshold <= 1.0, "
                f"got warn={warn_threshold}, critical={critical_threshold}."
            )
        self._warn_threshold = warn_threshold
        self._critical_threshold = critical_threshold
        self._queues: dict[str, QueueMetrics] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Queue registration
    # ------------------------------------------------------------------

    def register_queue(
        self,
        name: str,
        *,
        max_depth: int = 0,
        weight: float = 1.0,
    ) -> None:
        """Register a queue for pressure monitoring.

        Re-registering an existing name overwrites the previous entry.

        Parameters
        ----------
        name:
            Logical identifier for this queue.
        max_depth:
            Maximum queue capacity.  0 = unbounded (no pressure contribution).
        weight:
            Relative contribution to the aggregate pressure score.

        Raises
        ------
        ValueError
            If ``weight`` is negative or ``max_depth`` is negative.
        """
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}.")
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth}.")

        with self._lock:
            self._queues[name] = QueueMetrics(
                name=name,
                max_depth=max_depth,
                weight=weight,
            )
        logger.debug("BackpressureMonitor: registered queue %r.", name)

    def deregister_queue(self, name: str) -> None:
        """Remove a queue from monitoring.

        Silently ignores names that are not registered.
        """
        with self._lock:
            self._queues.pop(name, None)

    # ------------------------------------------------------------------
    # Depth updates
    # ------------------------------------------------------------------

    def update_depth(self, name: str, depth: int) -> None:
        """Update the current item count for a named queue.

        Parameters
        ----------
        name:
            The queue name as registered via ``register_queue``.
        depth:
            The current number of items in the queue.

        Raises
        ------
        KeyError
            If ``name`` has not been registered.
        ValueError
            If ``depth`` is negative.
        """
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}.")

        with self._lock:
            if name not in self._queues:
                raise KeyError(f"Queue {name!r} is not registered.")
            self._queues[name].current_depth = depth

    def update_depths(self, depths: dict[str, int]) -> None:
        """Bulk-update depths for multiple queues atomically.

        Parameters
        ----------
        depths:
            Mapping of queue name → current depth.  Unregistered names
            are silently skipped.
        """
        with self._lock:
            for name, depth in depths.items():
                if name in self._queues:
                    self._queues[name].current_depth = max(0, depth)

    # ------------------------------------------------------------------
    # Pressure queries
    # ------------------------------------------------------------------

    def pressure_score(self) -> float:
        """Return the weighted average pressure score across all queues.

        Returns
        -------
        float
            Value in [0.0, 1.0].  0.0 means all queues are empty;
            1.0 means all queues are completely full.
            Returns 0.0 if no queues are registered.
        """
        with self._lock:
            queues = list(self._queues.values())

        if not queues:
            return 0.0

        total_weight = sum(q.weight for q in queues)
        if total_weight == 0.0:
            return 0.0

        weighted_sum = sum(q.utilization * q.weight for q in queues)
        return weighted_sum / total_weight

    def is_warn(self) -> bool:
        """Return True when pressure_score >= warn_threshold."""
        return self.pressure_score() >= self._warn_threshold

    def is_critical(self) -> bool:
        """Return True when pressure_score >= critical_threshold."""
        return self.pressure_score() >= self._critical_threshold

    def queue_metrics(self, name: str) -> QueueMetrics:
        """Return a copy of metrics for the named queue.

        Raises
        ------
        KeyError
            If the queue is not registered.
        """
        with self._lock:
            if name not in self._queues:
                raise KeyError(f"Queue {name!r} is not registered.")
            metrics = self._queues[name]
            return QueueMetrics(
                name=metrics.name,
                current_depth=metrics.current_depth,
                max_depth=metrics.max_depth,
                weight=metrics.weight,
            )

    def all_metrics(self) -> list[QueueMetrics]:
        """Return a snapshot of metrics for all registered queues.

        Returns
        -------
        list[QueueMetrics]
            One entry per registered queue.
        """
        with self._lock:
            return [
                QueueMetrics(
                    name=q.name,
                    current_depth=q.current_depth,
                    max_depth=q.max_depth,
                    weight=q.weight,
                )
                for q in self._queues.values()
            ]

    @property
    def warn_threshold(self) -> float:
        """Configured warn threshold."""
        return self._warn_threshold

    @property
    def critical_threshold(self) -> float:
        """Configured critical threshold."""
        return self._critical_threshold

    def __repr__(self) -> str:
        score = self.pressure_score()
        return (
            f"BackpressureMonitor("
            f"queues={len(self._queues)}, "
            f"pressure={score:.2f}, "
            f"warn={self._warn_threshold}, "
            f"critical={self._critical_threshold}"
            f")"
        )
