"""PriorityMessageQueue — asyncio priority queue for MessageEnvelopes.

Messages with lower ``Priority`` integer values (higher urgency) are
dequeued first, giving CRITICAL (0) messages immediate precedence over
BATCH (4) messages.

The queue also enforces a configurable maximum depth, refusing new
messages when full and providing backpressure signals to producers.

Example
-------
::

    import asyncio
    from agent_mesh_router.broker.queue import PriorityMessageQueue
    from agent_mesh_router.messages.envelope import MessageEnvelope
    from agent_mesh_router.messages.types import Priority

    async def main():
        queue = PriorityMessageQueue(maxsize=100)

        envelope = MessageEnvelope(
            sender="a",
            receiver="b",
            payload={"task": "run"},
            priority=Priority.HIGH,
        )
        await queue.put(envelope)
        received = await queue.get()
        assert received.message_id == envelope.message_id

    asyncio.run(main())
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.types import Priority


@dataclass(order=True)
class _QueueItem:
    """Internal wrapper that makes MessageEnvelope sortable by priority."""

    priority_value: int
    sequence: int
    enqueued_at: float = field(compare=False)
    envelope: MessageEnvelope = field(compare=False)


class QueueFullError(RuntimeError):
    """Raised by ``put_nowait`` when the queue has reached its maximum depth."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(
            f"PriorityMessageQueue is full (maxsize={maxsize}). "
            "Apply backpressure or increase maxsize."
        )


class PriorityMessageQueue:
    """Async priority queue for ``MessageEnvelope`` objects.

    Messages are ordered by:
    1. ``Priority`` integer value (lower = higher urgency, dequeued first).
    2. Arrival sequence number (FIFO within the same priority level).

    Parameters
    ----------
    maxsize:
        Maximum number of messages the queue will hold.
        0 means unbounded (not recommended for production).
    discard_expired:
        When True, expired messages (TTL elapsed) are silently dropped
        at dequeue time rather than being returned to the consumer.
    """

    def __init__(
        self,
        *,
        maxsize: int = 0,
        discard_expired: bool = True,
    ) -> None:
        self._maxsize = maxsize
        self._discard_expired = discard_expired
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._sequence: int = 0
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._total_discarded: int = 0

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    async def put(self, envelope: MessageEnvelope) -> None:
        """Enqueue ``envelope``, waiting if the queue is full.

        Parameters
        ----------
        envelope:
            The envelope to enqueue.
        """
        item = self._make_item(envelope)
        await self._queue.put(item)
        self._total_enqueued += 1

    def put_nowait(self, envelope: MessageEnvelope) -> None:
        """Enqueue ``envelope`` without blocking.

        Raises
        ------
        QueueFullError
            If the queue is already at ``maxsize``.
        """
        item = self._make_item(envelope)
        try:
            self._queue.put_nowait(item)
            self._total_enqueued += 1
        except asyncio.QueueFull as exc:
            raise QueueFullError(self._maxsize) from exc

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    async def get(self) -> MessageEnvelope:
        """Dequeue and return the highest-priority envelope.

        Expired messages are skipped when ``discard_expired=True``.
        This method will block until a valid (non-expired) envelope is
        available.

        Returns
        -------
        MessageEnvelope
            The highest-priority non-expired envelope.
        """
        while True:
            item = await self._queue.get()
            self._total_dequeued += 1
            if self._discard_expired and item.envelope.is_expired():
                self._total_discarded += 1
                self._queue.task_done()
                continue
            return item.envelope

    def get_nowait(self) -> MessageEnvelope:
        """Dequeue without blocking.

        Raises
        ------
        asyncio.QueueEmpty
            If the queue is empty.
        """
        item = self._queue.get_nowait()
        self._total_dequeued += 1
        return item.envelope

    def task_done(self) -> None:
        """Signal that the most recently dequeued envelope was processed.

        Must be called once per ``get()`` or ``get_nowait()`` when using
        the ``join()`` completion guarantee.
        """
        self._queue.task_done()

    async def join(self) -> None:
        """Block until all enqueued items have been acknowledged via ``task_done``."""
        await self._queue.join()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def qsize(self) -> int:
        """Return the current number of messages in the queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if the queue contains no messages."""
        return self._queue.empty()

    def full(self) -> bool:
        """Return True if the queue has reached its maxsize."""
        return self._queue.full()

    @property
    def maxsize(self) -> int:
        """Configured maximum queue depth (0 = unbounded)."""
        return self._maxsize

    @property
    def total_enqueued(self) -> int:
        """Total messages added to the queue over its lifetime."""
        return self._total_enqueued

    @property
    def total_dequeued(self) -> int:
        """Total messages successfully dequeued over the queue's lifetime."""
        return self._total_dequeued

    @property
    def total_discarded(self) -> int:
        """Total messages discarded due to TTL expiry."""
        return self._total_discarded

    def __repr__(self) -> str:
        return (
            f"PriorityMessageQueue("
            f"size={self.qsize()}/{self._maxsize or '∞'}, "
            f"enqueued={self._total_enqueued}, "
            f"dequeued={self._total_dequeued}"
            f")"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_item(self, envelope: MessageEnvelope) -> _QueueItem:
        seq = self._sequence
        self._sequence += 1
        return _QueueItem(
            priority_value=int(envelope.priority),
            sequence=seq,
            enqueued_at=time.time(),
            envelope=envelope,
        )
