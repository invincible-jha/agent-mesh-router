"""Messages subsystem for agent-mesh-router.

Provides message types, envelopes, priorities, and serialization
for all inter-agent communication.

Example
-------
::

    from agent_mesh_router.messages import (
        MessageEnvelope,
        MessageType,
        Priority,
        HandoffMetrics,
        MessageSerializer,
    )

    envelope = MessageEnvelope(
        sender="agent-a",
        receiver="agent-b",
        payload={"task": "summarize", "text": "..."},
        message_type=MessageType.TASK,
        priority=Priority.HIGH,
    )
    envelope.validate()
    json_bytes = envelope.to_json()
    restored = MessageEnvelope.from_json(json_bytes)
"""
from __future__ import annotations

from agent_mesh_router.messages.envelope import MessageEnvelope
from agent_mesh_router.messages.serializer import MessageSerializer
from agent_mesh_router.messages.types import HandoffMetrics, MessageType, Priority

__all__ = [
    "MessageEnvelope",
    "MessageSerializer",
    "MessageType",
    "Priority",
    "HandoffMetrics",
]
