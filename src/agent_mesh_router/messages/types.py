"""Core message types, priority levels, and handoff metrics.

This module defines the foundational enumerations and dataclasses shared
across the entire agent-mesh-router message subsystem.

MessageType
-----------
Ten distinct message types cover all inter-agent communication patterns:
TASK, QUERY, RESPONSE, RESULT, HANDOFF, BROADCAST, HEARTBEAT, ERROR,
CANCEL, ACK.

Priority
--------
Five priority levels (CRITICAL=0 … LOW=4) align with asyncio.PriorityQueue
semantics — lower integer value means higher priority.

HandoffMetrics
--------------
Metadata captured when an agent hands off a task to a peer, enabling
cost accounting and latency tracking across workflow hops.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum


class MessageType(str, Enum):
    """Discriminator for inter-agent message semantics.

    Values are lowercase strings so they survive JSON round-trips
    as human-readable identifiers.
    """

    TASK = "task"
    """Delegate a new unit of work to a receiving agent."""

    QUERY = "query"
    """Ask the receiving agent for information without side-effects."""

    RESPONSE = "response"
    """Direct reply to a QUERY."""

    RESULT = "result"
    """Final output of a completed TASK."""

    HANDOFF = "handoff"
    """Transfer ownership of an in-progress task to another agent."""

    BROADCAST = "broadcast"
    """Publish a message to all subscribers on a topic."""

    HEARTBEAT = "heartbeat"
    """Liveness signal sent by an agent to the fleet registry."""

    ERROR = "error"
    """Notify the mesh of a processing failure."""

    CANCEL = "cancel"
    """Request cancellation of a previously dispatched message."""

    ACK = "ack"
    """Acknowledge receipt and acceptance of a message."""


class Priority(IntEnum):
    """Message priority levels, compatible with asyncio.PriorityQueue ordering.

    Lower integer values are processed first.

    Levels
    ------
    CRITICAL (0) — bypass normal queuing; must be processed immediately.
    HIGH (1)     — time-sensitive work, processed before NORMAL.
    NORMAL (2)   — default priority for most messages.
    LOW (3)      — background tasks, best-effort delivery.
    BATCH (4)    — bulk/offline processing, no latency guarantees.
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4


@dataclass
class HandoffMetrics:
    """Metadata captured when one agent hands a task to another.

    Attributes
    ----------
    source_agent:
        ID of the agent that is handing off the task.
    target_agent:
        ID of the agent receiving the task.
    reason:
        Human-readable explanation for why the handoff occurred
        (e.g. "capability_mismatch", "load_balancing", "cost_limit").
    cost_incurred_usd:
        USD cost accumulated by the source agent before handoff.
    tokens_consumed:
        Total LLM tokens consumed by the source agent.
    latency_ms:
        Wall-clock milliseconds elapsed at the source agent before handoff.
    handoff_timestamp:
        Unix epoch seconds (float) when the handoff was initiated.
        Defaults to ``time.time()`` at construction.
    attempt_number:
        How many times this task has been handed off (1-based counter).
    """

    source_agent: str
    target_agent: str
    reason: str
    cost_incurred_usd: float = 0.0
    tokens_consumed: int = 0
    latency_ms: float = 0.0
    handoff_timestamp: float = field(default_factory=time.time)
    attempt_number: int = 1

    def total_cost_label(self) -> str:
        """Return a formatted cost string suitable for display in logs."""
        return f"${self.cost_incurred_usd:.6f} USD"
