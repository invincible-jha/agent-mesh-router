"""A2A v0.3 Protocol Models.

Pydantic models for the Agent-to-Agent (A2A) open protocol defined by Google.
These models represent the core data structures for agent discovery, task
lifecycle management, and message exchange.

References
----------
- A2A Protocol specification v0.3

All fields use snake_case in Python; JSON serialization uses the same
snake_case keys to match the A2A spec wire format.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskState(str, Enum):
    """Valid states in the A2A task lifecycle state machine."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AgentSkill(BaseModel):
    """A discrete capability that an agent can perform.

    Parameters
    ----------
    id:
        Unique identifier for the skill within the agent.
    name:
        Human-readable display name.
    description:
        Explanation of what the skill does.
    tags:
        Searchable tags for discovery and filtering.
    examples:
        Example inputs or usage strings for documentation.
    """

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCapabilities(BaseModel):
    """Feature flags indicating which optional A2A capabilities an agent supports.

    Parameters
    ----------
    streaming:
        Whether the agent supports streaming task output.
    push_notifications:
        Whether the agent can push task status updates via webhooks.
    state_transition_history:
        Whether the agent maintains a full history of state transitions.
    """

    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False


class AgentCard(BaseModel):
    """A2A Agent Card — the machine-readable identity document for an agent.

    Agent Cards are served at ``/.well-known/agent.json`` and allow other
    agents and orchestration systems to discover capabilities without
    prior configuration.

    Parameters
    ----------
    name:
        Display name of the agent.
    description:
        Human-readable description of what the agent does.
    url:
        Base URL where the agent's A2A endpoint is reachable.
    version:
        A2A protocol version this agent implements.
    capabilities:
        Optional capability flags.
    skills:
        List of skills this agent can perform.
    default_input_modes:
        MIME types the agent accepts as input (e.g. ``text/plain``).
    default_output_modes:
        MIME types the agent produces as output.
    """

    name: str
    description: str
    url: str
    version: str = "0.3"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])


class MessagePart(BaseModel):
    """A single content part within an A2A message.

    Parameters
    ----------
    type:
        Content type identifier, e.g. ``"text"`` or ``"data"``.
    text:
        Text content when ``type`` is ``"text"``.
    """

    type: str = "text"
    text: str = ""


class A2AMessage(BaseModel):
    """A message exchanged between user and agent within a task.

    Parameters
    ----------
    role:
        Either ``"user"`` (input from the caller) or ``"agent"`` (output
        from the agent).
    parts:
        List of content parts comprising the message body.
    metadata:
        Arbitrary key-value metadata attached to the message.
    """

    role: str
    parts: list[MessagePart]
    metadata: dict[str, object] = Field(default_factory=dict)


class Artifact(BaseModel):
    """A discrete output artifact produced by an agent during task execution.

    Parameters
    ----------
    name:
        Identifier name for this artifact.
    description:
        Human-readable description of the artifact's content.
    parts:
        Content parts that make up the artifact.
    index:
        Zero-based ordering index when multiple artifacts are produced.
    """

    name: str
    description: str = ""
    parts: list[MessagePart]
    index: int = 0


class TaskStatus(BaseModel):
    """Point-in-time snapshot of a task's state.

    Parameters
    ----------
    state:
        Current lifecycle state.
    message:
        Optional message accompanying the state (e.g. progress update or
        final response).
    timestamp:
        UTC datetime when this status was recorded.
    """

    state: TaskState
    message: Optional[A2AMessage] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class A2ATask(BaseModel):
    """The full representation of an A2A task, including its history.

    Parameters
    ----------
    id:
        UUID identifying this task instance.
    status:
        The current status snapshot.
    artifacts:
        Artifacts produced during task execution.
    history:
        Ordered list of prior status snapshots (oldest first).
    metadata:
        Arbitrary key-value metadata associated with the task.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[TaskStatus] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
