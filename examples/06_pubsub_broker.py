#!/usr/bin/env python3
"""Example: Pub/Sub Broker and Priority Queue

Demonstrates topic-based publish/subscribe messaging and the
priority message queue for high-throughput agent communication.

Usage:
    python examples/06_pubsub_broker.py

Requirements:
    pip install agent-mesh-router
"""
from __future__ import annotations

import asyncio

import agent_mesh_router
from agent_mesh_router import (
    AsyncMessageBroker,
    MessageEnvelope,
    MessageType,
    Priority,
    PriorityMessageQueue,
    TopicManager,
)


async def demo_pubsub(topic_manager: TopicManager) -> None:
    """Publish messages and collect from subscribed handlers."""
    received: list[str] = []

    async def on_message(envelope: MessageEnvelope) -> None:
        received.append(str(envelope.payload.get("text", "")))

    sub = topic_manager.subscribe(topic="analysis-results", handler=on_message)
    print(f"Subscribed to 'analysis-results': sub_id={sub.subscription_id}")

    for i in range(3):
        env = MessageEnvelope(
            sender_id="analyser-agent",
            recipient_id="",
            message_type=MessageType.EVENT,
            priority=Priority.NORMAL,
            payload={"text": f"result-{i}"},
            topic="analysis-results",
        )
        await topic_manager.publish(env)

    print(f"Published 3 messages, received {len(received)}: {received}")

    topic_manager.unsubscribe(sub.subscription_id)


async def demo_broker() -> None:
    """Use AsyncMessageBroker to send and receive messages."""
    broker = AsyncMessageBroker()

    results: list[object] = []

    async def handler(envelope: MessageEnvelope) -> None:
        results.append(envelope.payload)

    broker.subscribe(recipient_id="worker-agent", handler=handler)

    envelope = MessageEnvelope(
        sender_id="orchestrator",
        recipient_id="worker-agent",
        message_type=MessageType.TASK,
        priority=Priority.HIGH,
        payload={"task": "process batch 1"},
    )
    await broker.send(envelope)
    print(f"Broker: delivered to worker-agent, payload={results[0]}")


def demo_priority_queue() -> None:
    """Enqueue messages at different priorities and dequeue in order."""
    queue: PriorityMessageQueue = PriorityMessageQueue(maxsize=100)

    for priority, text in [
        (Priority.LOW, "low-priority task"),
        (Priority.CRITICAL, "critical alert"),
        (Priority.NORMAL, "standard request"),
        (Priority.HIGH, "high-priority job"),
    ]:
        env = MessageEnvelope(
            sender_id="producer",
            recipient_id="consumer",
            message_type=MessageType.TASK,
            priority=priority,
            payload={"text": text},
        )
        queue.put(env)

    print(f"\nPriority queue: {queue.qsize()} messages")
    while not queue.empty():
        msg = queue.get()
        print(f"  [{msg.priority.value}] {msg.payload['text']}")


def main() -> None:
    print(f"agent-mesh-router version: {agent_mesh_router.__version__}")

    topic_manager = TopicManager()
    asyncio.run(demo_pubsub(topic_manager))
    asyncio.run(demo_broker())
    demo_priority_queue()


if __name__ == "__main__":
    main()
