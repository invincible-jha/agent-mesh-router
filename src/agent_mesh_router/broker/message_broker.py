"""AsyncMessageBroker — in-memory pub/sub with topic routing and TTL.

The broker is the central nervous system of an agent mesh.  Agents publish
messages through it; the broker delivers them to registered subscribers
(handlers) based on the envelope's ``receiver`` field and optional ``topic``.

Delivery semantics
------------------
* **Direct delivery**: if ``envelope.receiver`` is a specific agent ID that
  has a registered handler, the message is dispatched to that handler.
* **Topic delivery**: if ``envelope.topic`` is set, all subscribers to that
  topic receive a copy (fan-out), regardless of the ``receiver`` field.
* **Broadcast**: ``receiver="*"`` delivers to every registered handler.
* **TTL expiry**: expired envelopes are dead-lettered rather than dispatched.
* **Acknowledgement**: handlers that return successfully trigger ACK creation;
  exceptions trigger dead-lettering with ``"handler_error"`` reason.

Example
-------
::

    import asyncio
    from agent_mesh_router.broker.message_broker import AsyncMessageBroker
    from agent_mesh_router.messages.envelope import MessageEnvelope
    from agent_mesh_router.messages.types import MessageType, Priority

    async def main():
        broker = AsyncMessageBroker()
        await broker.start()

        received: list[MessageEnvelope] = []

        async def my_handler(env: MessageEnvelope) -> None:
            received.append(env)

        await broker.subscribe("agent-b", my_handler)

        env = MessageEnvelope(
            sender="agent-a",
            receiver="agent-b",
            payload={"cmd": "hello"},
        )
        await broker.publish(env)
        await asyncio.sleep(0.05)  # let dispatch loop process

        await broker.stop()

    asyncio.run(main())
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Union

from agent_mesh_router.broker.dead_letter import DeadLetterQueue
from agent_mesh_router.broker.queue import PriorityMessageQueue
from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.types import MessageType, Priority

logger = logging.getLogger(__name__)

MessageHandler = Callable[[MessageEnvelope], Union[Awaitable[None], None]]


class AsyncMessageBroker:
    """Async in-memory message broker with pub/sub and priority queuing.

    Parameters
    ----------
    queue_maxsize:
        Maximum depth for the internal priority queue.  0 = unbounded.
    dlq_maxsize:
        Maximum entries the dead-letter queue will retain.
    max_delivery_attempts:
        Number of times delivery will be retried before dead-lettering.
    dispatch_workers:
        Number of concurrent asyncio tasks consuming the internal queue.
    discard_expired:
        When True, TTL-expired messages are dead-lettered, not dispatched.
    """

    def __init__(
        self,
        *,
        queue_maxsize: int = 0,
        dlq_maxsize: int = 1000,
        max_delivery_attempts: int = 3,
        dispatch_workers: int = 4,
        discard_expired: bool = True,
    ) -> None:
        self._queue = PriorityMessageQueue(
            maxsize=queue_maxsize, discard_expired=discard_expired
        )
        self._dlq = DeadLetterQueue(maxsize=dlq_maxsize)
        self._max_delivery_attempts = max_delivery_attempts
        self._dispatch_workers = dispatch_workers

        # agent_id → handler
        self._agent_handlers: dict[str, MessageHandler] = {}
        # topic → set of agent_ids subscribed
        self._topic_subscribers: dict[str, set[str]] = defaultdict(set)

        self._running: bool = False
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the broker's dispatch worker tasks.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._running:
            return
        self._running = True
        self._worker_tasks = [
            asyncio.create_task(
                self._dispatch_loop(), name=f"broker-worker-{i}"
            )
            for i in range(self._dispatch_workers)
        ]
        logger.info(
            "AsyncMessageBroker started with %d workers.", self._dispatch_workers
        )

    async def stop(self) -> None:
        """Gracefully stop the broker.

        Waits for the internal queue to drain (up to a short timeout) before
        cancelling worker tasks.
        """
        if not self._running:
            return
        self._running = False

        # Give workers a chance to drain the queue.
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Broker stopped with messages still in queue.")

        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("AsyncMessageBroker stopped.")

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        agent_id: str,
        handler: MessageHandler,
    ) -> None:
        """Register a handler for direct messages to ``agent_id``.

        Parameters
        ----------
        agent_id:
            The agent identifier.  Must be unique; re-registering replaces
            the previous handler.
        handler:
            An async or sync callable that accepts a ``MessageEnvelope``.
        """
        async with self._lock:
            self._agent_handlers[agent_id] = handler
        logger.debug("Agent %r subscribed to broker.", agent_id)

    async def unsubscribe(self, agent_id: str) -> None:
        """Remove the handler for ``agent_id``.

        Also removes the agent from all topic subscriptions.

        Parameters
        ----------
        agent_id:
            Agent to unsubscribe.
        """
        async with self._lock:
            self._agent_handlers.pop(agent_id, None)
            for subscribers in self._topic_subscribers.values():
                subscribers.discard(agent_id)
        logger.debug("Agent %r unsubscribed from broker.", agent_id)

    async def subscribe_topic(self, agent_id: str, topic: str) -> None:
        """Subscribe ``agent_id`` to all messages published on ``topic``.

        The agent must first be subscribed via ``subscribe()``; topic
        subscriptions without a registered handler are silently ignored
        during dispatch.

        Parameters
        ----------
        agent_id:
            Agent to add to the topic subscriber set.
        topic:
            Topic string (e.g. ``"alerts"``, ``"task_updates"``).
        """
        async with self._lock:
            self._topic_subscribers[topic].add(agent_id)
        logger.debug("Agent %r subscribed to topic %r.", agent_id, topic)

    async def unsubscribe_topic(self, agent_id: str, topic: str) -> None:
        """Remove ``agent_id`` from the subscriber set for ``topic``.

        Parameters
        ----------
        agent_id:
            Agent to remove.
        topic:
            Topic string.
        """
        async with self._lock:
            self._topic_subscribers[topic].discard(agent_id)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, envelope: MessageEnvelope) -> None:
        """Enqueue ``envelope`` for asynchronous delivery.

        The broker does not deliver synchronously; call ``start()`` first to
        ensure dispatch workers are running.

        Parameters
        ----------
        envelope:
            The message to dispatch.
        """
        if envelope.is_expired():
            self._dlq.push(envelope, reason="ttl_expired_at_publish")
            logger.debug(
                "Message %s expired at publish time; dead-lettered.",
                envelope.message_id[:8],
            )
            return
        await self._queue.put(envelope)

    def publish_nowait(self, envelope: MessageEnvelope) -> None:
        """Enqueue without blocking (raises QueueFullError if queue is full).

        Parameters
        ----------
        envelope:
            The message to dispatch.
        """
        if envelope.is_expired():
            self._dlq.push(envelope, reason="ttl_expired_at_publish")
            return
        self._queue.put_nowait(envelope)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        """The broker's dead-letter queue for failed deliveries."""
        return self._dlq

    @property
    def queue(self) -> PriorityMessageQueue:
        """The broker's internal priority message queue."""
        return self._queue

    def subscriber_count(self) -> int:
        """Return the number of registered agent handlers."""
        return len(self._agent_handlers)

    def topic_subscriber_count(self, topic: str) -> int:
        """Return the number of agents subscribed to ``topic``."""
        return len(self._topic_subscribers.get(topic, set()))

    def is_running(self) -> bool:
        """Return True if the broker's dispatch workers are active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal dispatch loop
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Worker coroutine: dequeue and dispatch messages until stopped."""
        while self._running:
            try:
                try:
                    envelope = await asyncio.wait_for(
                        self._queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                await self._dispatch(envelope)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unhandled error in broker dispatch loop.")

    async def _dispatch(self, envelope: MessageEnvelope) -> None:
        """Deliver ``envelope`` to the appropriate handler(s)."""
        if envelope.is_expired():
            self._dlq.push(envelope, reason="ttl_expired_at_dispatch")
            return

        targets = await self._resolve_targets(envelope)

        if not targets:
            self._dlq.push(envelope, reason="no_subscriber")
            logger.debug(
                "No subscriber for message %s (receiver=%r, topic=%r).",
                envelope.message_id[:8],
                envelope.receiver,
                envelope.topic,
            )
            return

        for agent_id in targets:
            await self._deliver_to(envelope, agent_id)

    async def _resolve_targets(self, envelope: MessageEnvelope) -> list[str]:
        """Determine which agent IDs should receive the envelope."""
        targets: set[str] = set()

        async with self._lock:
            # Broadcast mode.
            if envelope.receiver == "*":
                targets.update(self._agent_handlers.keys())
            elif envelope.receiver in self._agent_handlers:
                targets.add(envelope.receiver)

            # Topic fan-out (in addition to direct delivery).
            if envelope.topic and envelope.topic in self._topic_subscribers:
                for subscriber_id in self._topic_subscribers[envelope.topic]:
                    if subscriber_id in self._agent_handlers:
                        targets.add(subscriber_id)

        return list(targets)

    async def _deliver_to(
        self,
        envelope: MessageEnvelope,
        agent_id: str,
        attempt: int = 1,
    ) -> None:
        """Invoke the handler for ``agent_id``, retrying on transient errors."""
        handler = self._agent_handlers.get(agent_id)
        if handler is None:
            self._dlq.push(
                envelope,
                reason="handler_disappeared",
                attempt_count=attempt,
            )
            return

        try:
            result = handler(envelope)
            if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            if attempt < self._max_delivery_attempts:
                logger.warning(
                    "Delivery attempt %d/%d failed for message %s → %r: %s",
                    attempt,
                    self._max_delivery_attempts,
                    envelope.message_id[:8],
                    agent_id,
                    exc,
                )
                await asyncio.sleep(0.05 * attempt)  # Small back-off.
                await self._deliver_to(envelope, agent_id, attempt + 1)
            else:
                logger.error(
                    "Message %s → %r dead-lettered after %d attempts: %s",
                    envelope.message_id[:8],
                    agent_id,
                    attempt,
                    exc,
                )
                self._dlq.push(
                    envelope,
                    reason="handler_error",
                    error_detail=str(exc),
                    attempt_count=attempt,
                )
