"""DeadLetterQueue — stores undeliverable messages for post-mortem inspection.

When the broker cannot deliver a message (no subscriber, TTL expired,
handler raised, delivery retry limit exceeded), it moves the envelope here
so operators can inspect failures without losing the data.

The DLQ is bounded.  When it fills up the oldest entries are evicted
(FIFO eviction) to make room for newer failures.

Example
-------
::

    from agent_mesh_router.broker.dead_letter import DeadLetterQueue, DeadLetterEntry

    dlq = DeadLetterQueue(maxsize=500)
    dlq.push(envelope, reason="no_subscriber")

    entries = dlq.list_entries()
    for entry in entries:
        print(entry.reason, entry.envelope.message_id)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


from agent_mesh_router.messages.envelope import MessageEnvelope


@dataclass
class DeadLetterEntry:
    """A single dead-letter record.

    Attributes
    ----------
    envelope:
        The message that could not be delivered.
    reason:
        Short machine-readable code describing why delivery failed
        (e.g. ``"no_subscriber"``, ``"ttl_expired"``, ``"handler_error"``).
    error_detail:
        Optional longer description of the failure (exception message, etc.).
    dead_lettered_at:
        Unix epoch seconds when the entry was added to the DLQ.
    attempt_count:
        Number of delivery attempts made before giving up.
    """

    envelope: MessageEnvelope
    reason: str
    error_detail: str = ""
    dead_lettered_at: float = field(default_factory=time.time)
    attempt_count: int = 1


class DeadLetterQueue:
    """Thread-safe bounded queue for undeliverable messages.

    Parameters
    ----------
    maxsize:
        Maximum number of dead-letter entries to retain.  When the limit
        is reached the oldest entry is evicted.  Defaults to 1000.
    """

    def __init__(self, *, maxsize: int = 1000) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}.")
        self._maxsize = maxsize
        self._entries: deque[DeadLetterEntry] = deque()
        self._lock = threading.Lock()
        self._total_received: int = 0

    def push(
        self,
        envelope: MessageEnvelope,
        reason: str,
        *,
        error_detail: str = "",
        attempt_count: int = 1,
    ) -> DeadLetterEntry:
        """Add an envelope to the dead-letter queue.

        If the queue is full, the oldest entry is evicted to make room.

        Parameters
        ----------
        envelope:
            The undeliverable message.
        reason:
            Short failure code (``"no_subscriber"``, ``"ttl_expired"``, etc.).
        error_detail:
            Longer description or exception text.
        attempt_count:
            Delivery attempts made before dead-lettering.

        Returns
        -------
        DeadLetterEntry
            The entry that was created and stored.
        """
        entry = DeadLetterEntry(
            envelope=envelope,
            reason=reason,
            error_detail=error_detail,
            attempt_count=attempt_count,
        )
        with self._lock:
            if len(self._entries) >= self._maxsize:
                self._entries.popleft()  # Evict oldest.
            self._entries.append(entry)
            self._total_received += 1
        return entry

    def pop(self) -> DeadLetterEntry | None:
        """Remove and return the oldest dead-letter entry, or None if empty.

        Returns
        -------
        DeadLetterEntry | None
            The oldest entry, or None when the queue is empty.
        """
        with self._lock:
            return self._entries.popleft() if self._entries else None

    def list_entries(
        self,
        *,
        reason_filter: str | None = None,
        limit: int | None = None,
    ) -> list[DeadLetterEntry]:
        """Return a snapshot of dead-letter entries.

        Parameters
        ----------
        reason_filter:
            When set, only entries with a matching ``reason`` are returned.
        limit:
            Maximum number of entries to return (newest-first when limited).

        Returns
        -------
        list[DeadLetterEntry]
            Snapshot ordered oldest-first.
        """
        with self._lock:
            entries = list(self._entries)

        if reason_filter is not None:
            entries = [e for e in entries if e.reason == reason_filter]

        if limit is not None:
            entries = entries[-limit:]  # Newest last → take tail.

        return entries

    def clear(self) -> int:
        """Remove all entries and return the count that was cleared.

        Returns
        -------
        int
            Number of entries removed.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def maxsize(self) -> int:
        """Configured maximum number of entries."""
        return self._maxsize

    @property
    def total_received(self) -> int:
        """Total entries ever pushed (including evicted ones)."""
        return self._total_received

    def is_empty(self) -> bool:
        """Return True if there are no dead-letter entries."""
        with self._lock:
            return len(self._entries) == 0

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"DeadLetterQueue(size={len(self._entries)}/{self._maxsize}, "
                f"total_received={self._total_received})"
            )
