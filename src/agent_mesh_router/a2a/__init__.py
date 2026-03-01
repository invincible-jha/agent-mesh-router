"""agent_mesh_router.a2a — A2A v0.3 Protocol Compatibility Layer.

Provides models, task lifecycle management, agent card generation, and
bridge utilities for the Agent-to-Agent (A2A) open protocol.

Exports
-------
Models
    TaskState, AgentSkill, AgentCapabilities, AgentCard,
    MessagePart, A2AMessage, Artifact, TaskStatus, A2ATask

Agent Card
    AgentCardGenerator, AgentCardError

Task Lifecycle
    TaskManager, TaskNotFoundError, InvalidTaskTransitionError,
    _VALID_TRANSITIONS (informational)

Discovery
    DiscoveryEndpoint

Bridge
    ACPBridge, ACPBridgeError

Example
-------
::

    from agent_mesh_router.a2a import (
        AgentCard,
        AgentCardGenerator,
        TaskManager,
        TaskState,
        A2AMessage,
        MessagePart,
        DiscoveryEndpoint,
        ACPBridge,
    )

    # Build an agent card
    generator = AgentCardGenerator()
    card = generator.from_config({"name": "MyAgent", "description": "...", "url": "..."})

    # Serve it
    endpoint = DiscoveryEndpoint(card)
    status, headers, body = endpoint.handle_request("/.well-known/agent.json")

    # Manage tasks
    manager = TaskManager()
    message = A2AMessage(role="user", parts=[MessagePart(text="Hello")])
    task = manager.create_task(message)
    task = manager.transition(task.id, TaskState.WORKING)
    task = manager.transition(task.id, TaskState.COMPLETED)
"""
from __future__ import annotations

from agent_mesh_router.a2a.acp_bridge import ACPBridge, ACPBridgeError
from agent_mesh_router.a2a.agent_card import AgentCardError, AgentCardGenerator
from agent_mesh_router.a2a.discovery import DiscoveryEndpoint
from agent_mesh_router.a2a.models import (
    A2AMessage,
    A2ATask,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    MessagePart,
    TaskState,
    TaskStatus,
)
from agent_mesh_router.a2a.task_lifecycle import (
    InvalidTaskTransitionError,
    TaskManager,
    TaskNotFoundError,
)

__all__: list[str] = [
    # models
    "A2AMessage",
    "A2ATask",
    "AgentCapabilities",
    "AgentCard",
    "AgentSkill",
    "Artifact",
    "MessagePart",
    "TaskState",
    "TaskStatus",
    # agent card
    "AgentCardError",
    "AgentCardGenerator",
    # task lifecycle
    "InvalidTaskTransitionError",
    "TaskManager",
    "TaskNotFoundError",
    # discovery
    "DiscoveryEndpoint",
    # bridge
    "ACPBridge",
    "ACPBridgeError",
]
