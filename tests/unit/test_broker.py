"""Tests for broker: DeadLetterQueue, PriorityMessageQueue, TopicManager, AsyncMessageBroker."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_mesh_router.broker.dead_letter import DeadLetterEntry, DeadLetterQueue
from agent_mesh_router.broker.message_broker import AsyncMessageBroker
from agent_mesh_router.broker.pubsub import TopicManager, TopicSubscription
from agent_mesh_router.broker.queue import PriorityMessageQueue, QueueFullError
from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.types import Priority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _envelope(
    sender: str = "a",
    receiver: str = "b",
    priority: Priority = Priority.NORMAL,
    topic: str | None = None,
) -> MessageEnvelope:
    return MessageEnvelope(
        sender=sender,
        receiver=receiver,
        payload={"x": 1},
        priority=priority,
        topic=topic,
    )


# ---------------------------------------------------------------------------
# DeadLetterQueue
# ---------------------------------------------------------------------------


class TestDeadLetterQueue:
    def test_invalid_maxsize(self) -> None:
        with pytest.raises(ValueError):
            DeadLetterQueue(maxsize=0)

    def test_push_and_len(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        env = _envelope()
        dlq.push(env, reason="no_subscriber")
        assert len(dlq) == 1

    def test_push_evicts_oldest_when_full(self) -> None:
        dlq = DeadLetterQueue(maxsize=2)
        e1, e2, e3 = _envelope(), _envelope(), _envelope()
        dlq.push(e1, reason="r1")
        dlq.push(e2, reason="r2")
        dlq.push(e3, reason="r3")
        assert len(dlq) == 2
        assert dlq.total_received == 3

    def test_pop_returns_oldest(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        e1, e2 = _envelope(), _envelope()
        dlq.push(e1, reason="first")
        dlq.push(e2, reason="second")
        entry = dlq.pop()
        assert entry is not None
        assert entry.reason == "first"

    def test_pop_empty_returns_none(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        assert dlq.pop() is None

    def test_list_entries_no_filter(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        dlq.push(_envelope(), reason="a")
        dlq.push(_envelope(), reason="b")
        entries = dlq.list_entries()
        assert len(entries) == 2

    def test_list_entries_reason_filter(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        dlq.push(_envelope(), reason="ttl_expired")
        dlq.push(_envelope(), reason="no_subscriber")
        entries = dlq.list_entries(reason_filter="ttl_expired")
        assert len(entries) == 1
        assert entries[0].reason == "ttl_expired"

    def test_list_entries_limit(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        for _ in range(5):
            dlq.push(_envelope(), reason="x")
        entries = dlq.list_entries(limit=3)
        assert len(entries) == 3

    def test_clear(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        dlq.push(_envelope(), reason="x")
        dlq.push(_envelope(), reason="x")
        removed = dlq.clear()
        assert removed == 2
        assert len(dlq) == 0

    def test_is_empty(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        assert dlq.is_empty() is True
        dlq.push(_envelope(), reason="x")
        assert dlq.is_empty() is False

    def test_maxsize_property(self) -> None:
        dlq = DeadLetterQueue(maxsize=42)
        assert dlq.maxsize == 42

    def test_repr(self) -> None:
        dlq = DeadLetterQueue(maxsize=5)
        r = repr(dlq)
        assert "DeadLetterQueue" in r

    def test_push_with_error_detail_and_attempt_count(self) -> None:
        dlq = DeadLetterQueue(maxsize=10)
        entry = dlq.push(_envelope(), reason="handler_error", error_detail="boom", attempt_count=3)
        assert entry.error_detail == "boom"
        assert entry.attempt_count == 3


# ---------------------------------------------------------------------------
# PriorityMessageQueue
# ---------------------------------------------------------------------------


class TestPriorityMessageQueue:
    @pytest.mark.asyncio
    async def test_put_and_get_ordering(self) -> None:
        q = PriorityMessageQueue()
        low = _envelope(priority=Priority.LOW)
        high = _envelope(priority=Priority.HIGH)
        await q.put(low)
        await q.put(high)
        first = await q.get()
        q.task_done()
        assert first.priority == Priority.HIGH

    @pytest.mark.asyncio
    async def test_put_nowait_raises_when_full(self) -> None:
        q = PriorityMessageQueue(maxsize=1)
        await q.put(_envelope())
        with pytest.raises(QueueFullError):
            q.put_nowait(_envelope())

    @pytest.mark.asyncio
    async def test_get_nowait_raises_when_empty(self) -> None:
        q = PriorityMessageQueue()
        with pytest.raises(asyncio.QueueEmpty):
            q.get_nowait()

    @pytest.mark.asyncio
    async def test_discard_expired(self) -> None:
        q = PriorityMessageQueue(discard_expired=True)
        env = MessageEnvelope(
            sender="a",
            receiver="b",
            payload={},
            ttl_seconds=0,  # Immediately expired.
        )
        # Give the TTL a moment to expire.
        import time
        time.sleep(0.01)
        await q.put(env)
        fresh = _envelope()
        await q.put(fresh)
        result = await asyncio.wait_for(q.get(), timeout=1.0)
        q.task_done()
        assert result.message_id == fresh.message_id
        assert q.total_discarded >= 1

    @pytest.mark.asyncio
    async def test_stats_counters(self) -> None:
        q = PriorityMessageQueue()
        await q.put(_envelope())
        await q.put(_envelope())
        assert q.total_enqueued == 2
        await q.get()
        q.task_done()
        assert q.total_dequeued == 1

    def test_qsize_empty_full(self) -> None:
        q = PriorityMessageQueue(maxsize=2)
        assert q.empty() is True
        assert q.full() is False
        assert q.maxsize == 2

    def test_repr(self) -> None:
        q = PriorityMessageQueue(maxsize=10)
        assert "PriorityMessageQueue" in repr(q)


# ---------------------------------------------------------------------------
# TopicManager
# ---------------------------------------------------------------------------


class TestTopicManager:
    def test_subscribe_and_publish(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "events.*")
        targets = m.publish("events.error", {"code": 500})
        assert "agent-a" in targets

    def test_exact_match_and_wildcard(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "events.*")
        m.subscribe("agent-b", "events.error")
        targets = m.publish("events.error", {})
        assert "agent-a" in targets
        assert "agent-b" in targets

    def test_no_match(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "metrics.*")
        targets = m.publish("events.error", {})
        assert targets == set()

    def test_unsubscribe_specific(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "events.*")
        m.subscribe("agent-b", "events.*")
        m.unsubscribe("agent-a", "events.*")
        targets = m.publish("events.info", {})
        assert "agent-a" not in targets
        assert "agent-b" in targets

    def test_unsubscribe_nonexistent_is_noop(self) -> None:
        m = TopicManager()
        m.unsubscribe("nobody", "nopattern")  # Should not raise.

    def test_unsubscribe_all(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "events.*")
        m.subscribe("agent-a", "metrics.*")
        removed = m.unsubscribe_all("agent-a")
        assert removed == 2
        assert m.publish("events.x", {}) == set()

    def test_unsubscribe_all_removes_empty_pattern(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "events.*")
        m.unsubscribe_all("agent-a")
        assert m.topic_count() == 0

    def test_subscribe_empty_agent_id_raises(self) -> None:
        m = TopicManager()
        with pytest.raises(ValueError):
            m.subscribe("", "topic.*")

    def test_subscribe_empty_pattern_raises(self) -> None:
        m = TopicManager()
        with pytest.raises(ValueError):
            m.subscribe("agent", "")

    def test_list_topics(self) -> None:
        m = TopicManager()
        m.subscribe("a", "b.c")
        m.subscribe("a", "a.*")
        topics = m.list_topics()
        assert sorted(topics) == topics  # Must be sorted.
        assert len(topics) == 2

    def test_list_subscribers(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "x")
        m.subscribe("agent-b", "x")
        subs = m.list_subscribers("x")
        assert subs == {"agent-a", "agent-b"}

    def test_list_agent_patterns(self) -> None:
        m = TopicManager()
        m.subscribe("agent-a", "x.*")
        m.subscribe("agent-a", "y.*")
        patterns = m.list_agent_patterns("agent-a")
        assert "x.*" in patterns
        assert "y.*" in patterns

    def test_resolve_subscribers_dry_run(self) -> None:
        m = TopicManager()
        m.subscribe("a", "t.*")
        subs = m.resolve_subscribers("t.hello")
        assert "a" in subs
        assert m.publish_count == 0  # Not incremented.

    def test_subscriber_count_all(self) -> None:
        m = TopicManager()
        m.subscribe("a", "x")
        m.subscribe("b", "x")
        assert m.subscriber_count() == 2

    def test_subscriber_count_by_pattern(self) -> None:
        m = TopicManager()
        m.subscribe("a", "x")
        m.subscribe("b", "y")
        assert m.subscriber_count("x") == 1

    def test_topic_count(self) -> None:
        m = TopicManager()
        m.subscribe("a", "x")
        m.subscribe("a", "y")
        assert m.topic_count() == 2

    def test_case_insensitive_mode(self) -> None:
        m = TopicManager(case_sensitive=False)
        m.subscribe("a", "Events.*")
        targets = m.publish("events.error", {})
        assert "a" in targets

    def test_counters_tracked(self) -> None:
        m = TopicManager()
        m.subscribe("a", "t")
        m.publish("t", {})
        m.publish("t", {})
        assert m.publish_count == 2
        assert m.total_deliveries == 2

    def test_repr(self) -> None:
        m = TopicManager()
        assert "TopicManager" in repr(m)


# ---------------------------------------------------------------------------
# AsyncMessageBroker
# ---------------------------------------------------------------------------


class TestAsyncMessageBroker:
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        assert broker.is_running() is True
        await broker.stop()
        assert broker.is_running() is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        await broker.start()  # Second call should be a no-op.
        assert broker.is_running()
        await broker.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_noop(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        await broker.stop()
        await broker.stop()  # Should not raise.

    @pytest.mark.asyncio
    async def test_subscribe_and_direct_delivery(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        received: list[MessageEnvelope] = []

        async def handler(env: MessageEnvelope) -> None:
            received.append(env)

        await broker.subscribe("agent-b", handler)
        env = _envelope(receiver="agent-b")
        await broker.publish(env)
        await asyncio.sleep(0.1)
        await broker.stop()
        assert len(received) == 1
        assert received[0].message_id == env.message_id

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_handler(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        received: list[MessageEnvelope] = []

        async def handler(env: MessageEnvelope) -> None:
            received.append(env)

        await broker.subscribe("agent-b", handler)
        await broker.unsubscribe("agent-b")
        env = _envelope(receiver="agent-b")
        await broker.publish(env)
        await asyncio.sleep(0.1)
        await broker.stop()
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_broadcast_delivery(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        received_a: list[MessageEnvelope] = []
        received_b: list[MessageEnvelope] = []

        await broker.subscribe("agent-a", lambda e: received_a.append(e))
        await broker.subscribe("agent-b", lambda e: received_b.append(e))

        env = _envelope(receiver="*")
        await broker.publish(env)
        await asyncio.sleep(0.2)
        await broker.stop()
        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_topic_delivery(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        received: list[MessageEnvelope] = []

        async def handler(env: MessageEnvelope) -> None:
            received.append(env)

        await broker.subscribe("agent-x", handler)
        await broker.subscribe_topic("agent-x", "alerts")
        env = _envelope(receiver="nobody", topic="alerts")
        await broker.publish(env)
        await asyncio.sleep(0.1)
        await broker.stop()
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_no_subscriber_dead_letters(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        env = _envelope(receiver="nonexistent-agent")
        await broker.publish(env)
        await asyncio.sleep(0.1)
        await broker.stop()
        assert len(broker.dead_letter_queue) >= 1

    @pytest.mark.asyncio
    async def test_handler_error_dead_letters_after_retries(self) -> None:
        broker = AsyncMessageBroker(max_delivery_attempts=2)
        await broker.start()

        async def failing_handler(env: MessageEnvelope) -> None:
            raise RuntimeError("Intentional failure")

        await broker.subscribe("agent-x", failing_handler)
        env = _envelope(receiver="agent-x")
        await broker.publish(env)
        await asyncio.sleep(0.5)
        await broker.stop()
        dlq_entries = broker.dead_letter_queue.list_entries(reason_filter="handler_error")
        assert len(dlq_entries) >= 1

    @pytest.mark.asyncio
    async def test_publish_nowait(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        received: list[MessageEnvelope] = []
        await broker.subscribe("b", lambda e: received.append(e))
        env = _envelope(receiver="b")
        broker.publish_nowait(env)
        await asyncio.sleep(0.1)
        await broker.stop()
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_expired_message_dead_lettered_at_publish(self) -> None:
        import time
        broker = AsyncMessageBroker()
        await broker.start()
        env = MessageEnvelope(
            sender="a", receiver="b", payload={}, ttl_seconds=0
        )
        time.sleep(0.01)
        await broker.publish(env)
        await asyncio.sleep(0.05)
        await broker.stop()
        entries = broker.dead_letter_queue.list_entries(reason_filter="ttl_expired_at_publish")
        assert len(entries) >= 1

    def test_subscriber_count(self) -> None:
        broker = AsyncMessageBroker()
        assert broker.subscriber_count() == 0

    def test_topic_subscriber_count(self) -> None:
        broker = AsyncMessageBroker()
        assert broker.topic_subscriber_count("alerts") == 0

    def test_queue_and_dlq_properties(self) -> None:
        broker = AsyncMessageBroker()
        assert broker.queue is not None
        assert broker.dead_letter_queue is not None

    @pytest.mark.asyncio
    async def test_unsubscribe_topic(self) -> None:
        broker = AsyncMessageBroker()
        await broker.start()
        received: list[MessageEnvelope] = []

        async def handler(env: MessageEnvelope) -> None:
            received.append(env)

        await broker.subscribe("agent-x", handler)
        await broker.subscribe_topic("agent-x", "alerts")
        await broker.unsubscribe_topic("agent-x", "alerts")

        env = _envelope(receiver="nobody", topic="alerts")
        await broker.publish(env)
        await asyncio.sleep(0.1)
        await broker.stop()
        assert len(received) == 0
