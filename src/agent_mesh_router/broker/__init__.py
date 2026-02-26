"""Message broker subsystem for agent-mesh-router.

Provides in-memory pub/sub with topic routing, priority queuing,
and dead-letter handling.

Example
-------
::

    import asyncio
    from agent_mesh_router.broker import AsyncMessageBroker

    async def main():
        broker = AsyncMessageBroker()
        await broker.start()

        async def handler(env):
            print(f"Received: {env.payload}")

        await broker.subscribe("agent-b", handler)
        await broker.publish(envelope)
        await broker.stop()

    asyncio.run(main())
"""
from __future__ import annotations

from agent_mesh_router.broker.dead_letter import DeadLetterQueue
from agent_mesh_router.broker.message_broker import AsyncMessageBroker
from agent_mesh_router.broker.queue import PriorityMessageQueue

__all__ = [
    "AsyncMessageBroker",
    "PriorityMessageQueue",
    "DeadLetterQueue",
]
