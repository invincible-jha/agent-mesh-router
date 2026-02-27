"""Backpressure-aware bounded task queue.

A thread-safe FIFO queue that integrates with :class:`BackpressureMonitor`
to apply automatic throttling when system pressure rises above configurable
watermarks.

Design
------
- Tasks are any callable that accepts no arguments.
- The queue has a configurable maximum capacity (*max_size*).  When the
  queue is full, :meth:`~BackpressureQueue.enqueue` blocks (blocking mode)
  or raises :class:`QueueFullError` (non-blocking mode).
- Two pressure watermarks gate enqueue behaviour:

  ``high_watermark``
      When the queue fill ratio exceeds this fraction, enqueue begins to
      emit warnings and the queue is considered "under pressure".
  ``low_watermark``
      Once pressure drops back below this fraction, normal operation resumes.

- The queue reports its depth to a :class:`~BackpressureMonitor` instance
  after every enqueue / dequeue so the monitor's pressure score stays fresh.

Usage
-----
::

    from agent_mesh_router.backpressure import BackpressureMonitor
    from agent_mesh_router.backpressure.task_queue import BackpressureQueue

    monitor = BackpressureMonitor()
    queue = BackpressureQueue(max_size=100, monitor=monitor, queue_name="main")

    queue.enqueue(lambda: print("task 1"))
    task = queue.dequeue()
    task()
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable

from agent_mesh_router.backpressure.monitor import BackpressureMonitor

logger = logging.getLogger(__name__)

Task = Callable[[], object]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QueueFullError(RuntimeError):
    """Raised when a non-blocking enqueue fails because the queue is full."""


class QueueEmptyError(RuntimeError):
    """Raised when a non-blocking dequeue is attempted on an empty queue."""


# ---------------------------------------------------------------------------
# QueueStats snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueStats:
    """Point-in-time snapshot of queue health.

    Parameters
    ----------
    size:
        Current number of tasks in the queue.
    max_size:
        Maximum queue capacity.
    fill_ratio:
        ``size / max_size`` in [0.0, 1.0].
    is_under_pressure:
        True when fill_ratio >= high_watermark.
    enqueued_total:
        Cumulative count of successfully enqueued tasks.
    dequeued_total:
        Cumulative count of successfully dequeued tasks.
    dropped_total:
        Cumulative count of tasks rejected due to a full queue.
    """

    size: int
    max_size: int
    fill_ratio: float
    is_under_pressure: bool
    enqueued_total: int
    dequeued_total: int
    dropped_total: int


# ---------------------------------------------------------------------------
# BackpressureQueue
# ---------------------------------------------------------------------------


class BackpressureQueue:
    """Bounded task queue with backpressure awareness.

    Parameters
    ----------
    max_size:
        Maximum number of tasks the queue can hold.  Must be >= 1.
    monitor:
        Optional :class:`BackpressureMonitor` to receive queue depth updates.
        If None a standalone monitor is created for internal use.
    queue_name:
        Logical name used when registering with *monitor*.
    high_watermark:
        Fill ratio at or above which the queue is considered "under pressure".
        Must be in (0.0, 1.0].  Defaults to 0.8.
    low_watermark:
        Fill ratio below which pressure is considered resolved.
        Must be in [0.0, *high_watermark*).  Defaults to 0.5.

    Example
    -------
    ::

        queue = BackpressureQueue(max_size=10)
        queue.enqueue(lambda: None)
        task = queue.dequeue()
    """

    def __init__(
        self,
        max_size: int = 100,
        monitor: BackpressureMonitor | None = None,
        queue_name: str = "backpressure_queue",
        high_watermark: float = 0.8,
        low_watermark: float = 0.5,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size!r}.")
        if not (0.0 < high_watermark <= 1.0):
            raise ValueError(
                f"high_watermark must be in (0.0, 1.0], got {high_watermark!r}."
            )
        if not (0.0 <= low_watermark < high_watermark):
            raise ValueError(
                f"low_watermark must be in [0.0, high_watermark), "
                f"got low={low_watermark!r}, high={high_watermark!r}."
            )

        self._max_size = max_size
        self._high_watermark = high_watermark
        self._low_watermark = low_watermark
        self._queue_name = queue_name

        # Bounded stdlib queue
        self._q: queue.Queue[Task] = queue.Queue(maxsize=max_size)

        # Counters
        self._lock = threading.Lock()
        self._enqueued_total: int = 0
        self._dequeued_total: int = 0
        self._dropped_total: int = 0
        self._under_pressure: bool = False

        # Monitor integration
        if monitor is None:
            monitor = BackpressureMonitor()
        self._monitor = monitor
        self._monitor.register_queue(queue_name, max_depth=max_size)

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, task: Task, block: bool = True, timeout: float | None = None) -> None:
        """Add *task* to the queue.

        Parameters
        ----------
        task:
            A zero-argument callable to enqueue.
        block:
            If True (default), block until space is available or *timeout*
            expires.  If False, raise :class:`QueueFullError` immediately
            when the queue is full.
        timeout:
            Maximum seconds to wait when ``block=True``.  None means wait
            indefinitely.

        Raises
        ------
        QueueFullError
            If the queue is full and either ``block=False`` or *timeout*
            expired.
        TypeError
            If *task* is not callable.
        """
        if not callable(task):
            raise TypeError(f"task must be callable, got {type(task).__name__!r}.")

        try:
            self._q.put(task, block=block, timeout=timeout)
        except queue.Full as exc:
            with self._lock:
                self._dropped_total += 1
            raise QueueFullError(
                f"Queue {self._queue_name!r} is full ({self._max_size} tasks)."
            ) from exc

        with self._lock:
            self._enqueued_total += 1
            current_size = self._q.qsize()
            fill = current_size / self._max_size
            if fill >= self._high_watermark and not self._under_pressure:
                self._under_pressure = True
                logger.warning(
                    "BackpressureQueue %r: high watermark reached (fill=%.2f).",
                    self._queue_name,
                    fill,
                )
            self._monitor.update_depth(self._queue_name, current_size)

    def enqueue_nowait(self, task: Task) -> None:
        """Non-blocking enqueue — shorthand for ``enqueue(task, block=False)``.

        Raises
        ------
        QueueFullError
            If the queue is full.
        """
        self.enqueue(task, block=False)

    # ------------------------------------------------------------------
    # Dequeue
    # ------------------------------------------------------------------

    def dequeue(self, block: bool = True, timeout: float | None = None) -> Task:
        """Remove and return the next task from the queue (FIFO).

        Parameters
        ----------
        block:
            If True (default), block until a task is available or *timeout*
            expires.  If False, raise :class:`QueueEmptyError` immediately
            when the queue is empty.
        timeout:
            Maximum seconds to wait when ``block=True``.  None means wait
            indefinitely.

        Returns
        -------
        Task
            The next callable task.

        Raises
        ------
        QueueEmptyError
            If the queue is empty and either ``block=False`` or *timeout*
            expired.
        """
        try:
            task = self._q.get(block=block, timeout=timeout)
        except queue.Empty as exc:
            raise QueueEmptyError(
                f"Queue {self._queue_name!r} is empty."
            ) from exc

        with self._lock:
            self._dequeued_total += 1
            current_size = self._q.qsize()
            fill = current_size / self._max_size
            if fill < self._low_watermark and self._under_pressure:
                self._under_pressure = False
                logger.info(
                    "BackpressureQueue %r: pressure relieved (fill=%.2f).",
                    self._queue_name,
                    fill,
                )
            self._monitor.update_depth(self._queue_name, current_size)

        return task

    def dequeue_nowait(self) -> Task:
        """Non-blocking dequeue — shorthand for ``dequeue(block=False)``.

        Raises
        ------
        QueueEmptyError
            If the queue is empty.
        """
        return self.dequeue(block=False)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of tasks in the queue."""
        return self._q.qsize()

    @property
    def max_size(self) -> int:
        """Maximum queue capacity."""
        return self._max_size

    @property
    def fill_ratio(self) -> float:
        """Current fill fraction in [0.0, 1.0]."""
        return self._q.qsize() / self._max_size

    @property
    def is_under_pressure(self) -> bool:
        """True when the fill ratio is at or above the high watermark."""
        with self._lock:
            return self._under_pressure

    @property
    def is_empty(self) -> bool:
        """True when the queue contains no tasks."""
        return self._q.empty()

    @property
    def is_full(self) -> bool:
        """True when the queue has reached max_size."""
        return self._q.full()

    @property
    def monitor(self) -> BackpressureMonitor:
        """The associated :class:`BackpressureMonitor`."""
        return self._monitor

    def stats(self) -> QueueStats:
        """Return a point-in-time :class:`QueueStats` snapshot.

        Returns
        -------
        QueueStats
            Current queue health metrics.
        """
        with self._lock:
            current_size = self._q.qsize()
            return QueueStats(
                size=current_size,
                max_size=self._max_size,
                fill_ratio=current_size / self._max_size,
                is_under_pressure=self._under_pressure,
                enqueued_total=self._enqueued_total,
                dequeued_total=self._dequeued_total,
                dropped_total=self._dropped_total,
            )

    def drain(self) -> list[Task]:
        """Remove and return all tasks currently in the queue.

        Returns
        -------
        list[Task]
            All tasks in FIFO order.  May be empty.
        """
        tasks: list[Task] = []
        while True:
            try:
                tasks.append(self._q.get_nowait())
            except queue.Empty:
                break
        with self._lock:
            self._dequeued_total += len(tasks)
            self._monitor.update_depth(self._queue_name, 0)
            if self._under_pressure:
                self._under_pressure = False
        return tasks

    def __len__(self) -> int:
        """Return the current number of tasks in the queue."""
        return self._q.qsize()

    def __repr__(self) -> str:
        return (
            f"BackpressureQueue("
            f"name={self._queue_name!r}, "
            f"size={self.size}/{self._max_size}, "
            f"pressure={self._under_pressure}"
            f")"
        )
